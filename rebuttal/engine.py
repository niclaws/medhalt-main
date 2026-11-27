"""Full-sequence, device-map-safe intervention engine for large checkpoints.
No modification to original engine. Avoids reconstructing Gemma sliding-window KV caches.
"""

from __future__ import annotations
from types import SimpleNamespace
import time
import numpy as np
import torch
from patching_engine import CausalEngine, ComponentSpace, vocab_stats
from .core import LOG
from ._timing import timed, timed_call


@timed
def resolve_quantization(tag, requested="auto", reuse_checkpoint=False):
    from .core import LARGE_MODELS

    mode = (
        ("nf4" if tag in LARGE_MODELS else "none") if requested == "auto" else requested
    )
    if mode not in ("none", "nf4"):
        raise ValueError(f"Unknown quantization: {mode}")
    if reuse_checkpoint and mode != "none":
        raise ValueError(
            "Quantized inference cannot reuse BF16 discovery checkpoints. Use a fresh scale run."
        )
    return mode


@timed
def nf4_load_kwargs(dtype, max_memory=None, allow_cpu_offload=False):
    """Explicit logical CUDA device 0; never silently spill to CPU or disk."""
    if allow_cpu_offload or max_memory is not None:
        raise ValueError(
            'NF4 single-GPU mode uses device_map="cuda"; omit --max-memory and --allow-cpu-offload.'
        )
    if dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("NF4 compute dtype must be bfloat16 or float16")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "NF4 requires exactly one visible CUDA GPU; set CUDA_VISIBLE_DEVICES to the allocated GPU."
        )
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "GPU does not support BF16; use --dtype float16 in a fresh run."
        )
    try:
        from transformers import BitsAndBytesConfig
        import bitsandbytes  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "Install rebuttal/requirements-large.txt in a CUDA-compatible environment."
        ) from e
    return dict(
        device_map="cuda",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        ),
    )


@timed
def verify_nf4_model(model):
    import bitsandbytes as bnb

    if not getattr(model, "is_loaded_in_4bit", False):
        raise RuntimeError("Model did not report successful 4-bit loading")
    dm = "auto"
    bad = [
        (n, str(p.device))
        for n, p in model.named_parameters()
        if p.device.type != "cuda" or p.device.index != 0
    ]
    if bad:
        raise RuntimeError(f"Parameters outside CUDA 0: {bad[:5]}")
    count = sum(isinstance(m, bnb.nn.Linear4bit) for m in model.modules())
    if not count:
        raise RuntimeError("No BitsAndBytes Linear4bit modules found")
    return count


class ShardedEngine(CausalEngine):
    @timed
    def __init__(
        self,
        model_name,
        dtype="bfloat16",
        trust_remote_code=False,
        attn_implementation="sdpa",
        max_memory=None,
        revision=None,
        allow_cpu_offload=False,
        quantization="none",
    ):
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.model_name = model_name
        self.dtype = getattr(torch, dtype)
        self.quantization = quantization
        quant_kw = (
            nf4_load_kwargs(self.dtype, max_memory, allow_cpu_offload)
            if quantization == "nf4"
            else {}
        )
        if quantization not in ("none", "nf4"):
            raise ValueError(f"Unsupported quantization: {quantization}")
        started = time.monotonic()
        kw = dict(trust_remote_code=trust_remote_code, revision=revision)
        self.tokenizer = timed_call(
            "AutoTokenizer.from_pretrained",
            AutoTokenizer.from_pretrained,
            model_name,
            **kw,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.has_chat_template = bool(self.tokenizer.chat_template)
        args = dict(
            **kw,
            torch_dtype=self.dtype,
            device_map="cuda",
            low_cpu_mem_usage=True,
        )
        if max_memory:
            args["max_memory"] = max_memory
        if attn_implementation:
            args["attn_implementation"] = attn_implementation
        args.update(quant_kw)
        self.model = timed_call(
            "AutoModelForCausalLM.from_pretrained",
            AutoModelForCausalLM.from_pretrained,
            model_name,
            **args,
        ).eval()
        self.model.requires_grad_(False)
        dm = "auto"
        self.quantization_metadata = dict(
            mode=quantization,
            compute_dtype=str(self.dtype),
            loaded_in_4bit=bool(getattr(self.model, "is_loaded_in_4bit", False)),
        )
        if quantization == "nf4":
            self.quantization_metadata.update(
                type="nf4",
                double_quant=True,
                linear4bit_modules=verify_nf4_model(self.model),
                single_gpu=True,
            )
        self.quantization_metadata["load_seconds"] = time.monotonic() - started
        if torch.cuda.is_available():
            self.quantization_metadata.update(
                gpu=torch.cuda.get_device_name(0),
                allocated_bytes=torch.cuda.memory_allocated(0),
                reserved_bytes=torch.cuda.memory_reserved(0),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
                free_bytes=torch.cuda.mem_get_info(0)[0],
            )
        LOG.info("Model loading: %s", self.quantization_metadata)
        LOG.info("Device map: %s", dm)
        self.device = self.model.get_input_embeddings().weight.device
        if self.device.type == "meta":
            raise RuntimeError("Embedding offload unsupported")
        self.layers = timed_call("self._find_layers", self._find_layers)
        cfg = getattr(self.model.config, "text_config", self.model.config)
        self.num_layers = len(self.layers)
        self.num_heads = int(cfg.num_attention_heads)
        self.hidden_size = int(cfg.hidden_size)
        self.head_dim = int(
            getattr(cfg, "head_dim", None) or self.hidden_size // self.num_heads
        )
        self.o_projs = [l.self_attn.o_proj for l in self.layers]
        self.mlps = [l.mlp for l in self.layers]
        for l, m in enumerate(self.o_projs):
            if m.in_features != self.num_heads * self.head_dim:
                raise ValueError(f"Unexpected head shape at layer {l}: {m.in_features}")
        self.space = ComponentSpace(self.num_layers, self.num_heads)
        LOG.info(
            "Loaded %s layers=%s heads=%s head_dim=%s hidden=%s",
            model_name,
            self.num_layers,
            self.num_heads,
            self.head_dim,
            self.hidden_size,
        )

    @timed
    def _forward(self, ids, batch):
        LOG.debug("Forward tokens=%d batch=%d", len(ids), batch)
        x = (
            torch.tensor([list(ids)], dtype=torch.long, device=self.device)
            .expand(batch, -1)
            .contiguous()
        )
        # Many HF models support logits_to_keep; inspect the actual signature instead of assuming.
        import inspect

        kw = {}
        params = inspect.signature(self.model.forward).parameters
        if "logits_to_keep" in params:
            kw["logits_to_keep"] = 1
        elif "num_logits_to_keep" in params:
            kw["num_logits_to_keep"] = 1
        return (
            timed_call(
                "self.model",
                self.model,
                input_ids=x,
                attention_mask=torch.ones_like(x),
                use_cache=False,
                **kw,
            )
            .logits[:, -1]
            .float()
        )

    @timed
    @torch.inference_mode()
    def prepare(self, input_ids, cand_ids):
        z = [None] * self.num_layers
        mlp = z.copy()
        resid = z.copy()
        hooks = []

        def pre(l):
            def f(m, a):
                z[l] = a[0][0, -1].detach().cpu().clone()

            return f

        def post(arr, l):
            def f(m, a, o):
                arr[l] = (
                    (o[0] if isinstance(o, tuple) else o)[0, -1].detach().cpu().clone()
                )

            return f

        try:
            for l in range(self.num_layers):
                hooks.extend(
                    [
                        self.o_projs[l].register_forward_pre_hook(pre(l)),
                        self.mlps[l].register_forward_hook(post(mlp, l)),
                        self.layers[l].register_forward_hook(post(resid, l)),
                    ]
                )
            LOG.debug("Activation capture hooks registered=%d", len(hooks))
            logits = self._forward(input_ids, 1)[0]
            vs = vocab_stats(logits)
            cand = torch.tensor(list(cand_ids), device=logits.device, dtype=torch.long)
            return SimpleNamespace(
                input_ids=list(input_ids),
                cand_ids=cand.cpu(),
                cand_logits=logits[cand].cpu(),
                vocab_stats=vs,
                vocab_entropy=vs[0],
                z=torch.stack(z),
                mlp=torch.stack(mlp),
                resid=torch.stack(resid),
            )
        finally:
            for h in hooks:
                h.remove()

    @timed
    def prompt_features(self, input_ids):
        s = self.prepare(input_ids, [0, 1])
        return s.vocab_stats, s.resid

    @timed
    @torch.inference_mode()
    def run_interventions(
        self,
        state,
        interventions,
        source=None,
        steer_z=None,
        steer_mlp=None,
        batch_size=1,
        steer_resid=None,
    ):
        if not len(interventions):
            return np.empty((0, len(state.cand_ids)), np.float32)
        if steer_resid is not None and steer_resid.ndim == 2:
            steer_resid = steer_resid.unsqueeze(0)
        out = []
        start = 0
        bs = min(batch_size, getattr(self, "_safe_batch", batch_size))
        started = time.monotonic()
        while start < len(interventions):
            chunk = interventions[start : start + bs]
            LOG.debug(
                "Intervention batch offset=%d count=%d total=%d batch_size=%d",
                start,
                len(chunk),
                len(interventions),
                bs,
            )
            try:
                out.append(
                    self._chunk(state, chunk, source, steer_z, steer_mlp, steer_resid)
                )
                start += len(chunk)
            except torch.cuda.OutOfMemoryError:
                if bs == 1:
                    LOG.error("CUDA OOM at minimum batch size offset=%d", start)
                    raise
                timed_call("torch.cuda.empty_cache", torch.cuda.empty_cache)
                bs = max(1, bs // 2)
                self._safe_batch = bs
                LOG.warning("OOM: reduced intervention batch to %s", bs)
        if getattr(self, "quantization", "none") == "nf4":
            self._perf_calls = getattr(self, "_perf_calls", 0) + 1
            self._perf_evals = getattr(self, "_perf_evals", 0) + len(interventions)
            self._perf_seconds = (
                getattr(self, "_perf_seconds", 0.0) + time.monotonic() - started
            )
            if self._perf_calls == 1 or self._perf_calls % 50 == 0:
                LOG.info(
                    "NF4 intervention throughput: calls=%d evaluations=%d seconds=%.2f eval/s=%.2f batch=%d peak_vram_GiB=%.2f",
                    self._perf_calls,
                    self._perf_evals,
                    self._perf_seconds,
                    self._perf_evals / max(self._perf_seconds, 1e-9),
                    bs,
                    torch.cuda.max_memory_allocated(0) / 2**30,
                )
        return np.concatenate(out)

    @timed
    def _chunk(self, state, ivs, source, vz, vm, vr):
        P, S, R, RS = timed_call("self._build_masks", self._build_masks, ivs)
        H = self.num_heads
        d = self.head_dim
        hooks = []

        def zhook(l):
            def f(m, args):
                x = args[0].clone()
                last = x[:, -1].reshape(len(ivs), H, d)
                dev = last.device
                if P[:, l, :H].any():
                    last = torch.where(
                        torch.as_tensor(P[:, l, :H], device=dev)[..., None],
                        source.z[l].to(dev, last.dtype).reshape(1, H, d),
                        last,
                    )
                if S[:, l, :H].any():
                    last = (
                        last.float()
                        + torch.as_tensor(S[:, l, :H], device=dev)[..., None]
                        * vz[l].to(dev, torch.float32).reshape(1, H, d)
                    ).to(x.dtype)
                x[:, -1] = last.reshape(len(ivs), -1)
                return (x,) + tuple(args[1:])

            return f

        def post(l, resid):
            def f(m, args, o):
                is_tuple = isinstance(o, tuple)
                x = (o[0] if is_tuple else o).clone()
                last = x[:, -1]
                dev = last.device
                if resid:
                    v = vr[torch.as_tensor(RS, device=vr.device), l].to(
                        dev, torch.float32
                    )
                    last = (
                        last.float() + torch.as_tensor(R[:, l], device=dev)[:, None] * v
                    ).to(x.dtype)
                else:
                    if P[:, l, H].any():
                        last = torch.where(
                            torch.as_tensor(P[:, l, H], device=dev)[:, None],
                            source.mlp[l].to(dev, last.dtype)[None],
                            last,
                        )
                    if S[:, l, H].any():
                        last = (
                            last.float()
                            + torch.as_tensor(S[:, l, H], device=dev)[:, None]
                            * vm[l].to(dev, torch.float32)[None]
                        ).to(x.dtype)
                x[:, -1] = last
                return (x,) + o[1:] if is_tuple else x

            return f

        try:
            for l in range(self.num_layers):
                if P[:, l, :H].any() or S[:, l, :H].any():
                    hooks.append(self.o_projs[l].register_forward_pre_hook(zhook(l)))
                if P[:, l, H].any() or S[:, l, H].any():
                    hooks.append(self.mlps[l].register_forward_hook(post(l, False)))
                if R[:, l].any():
                    hooks.append(self.layers[l].register_forward_hook(post(l, True)))
            LOG.debug("Intervention hooks registered=%d", len(hooks))
            logits = self._forward(state.input_ids, len(ivs))
            return logits[:, state.cand_ids.to(logits.device)].cpu().numpy()
        finally:
            for h in hooks:
                h.remove()
