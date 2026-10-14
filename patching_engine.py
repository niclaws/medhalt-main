from __future__ import annotations

import math
import numpy as np
import torch
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo


def now() -> str:
    return datetime.now()


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


# =====================================================================
# Component universe
# =====================================================================
# Components are indexed layer-major: for layer l, heads 0..H-1 then the MLP.
#   index = l * (H + 1) + h      (head h)
#   index = l * (H + 1) + H      (MLP of layer l)


@dataclass(frozen=True)
class ComponentSpace:
    num_layers: int
    num_heads: int

    @property
    def per_layer(self) -> int:
        return self.num_heads + 1

    @property
    def size(self) -> int:
        return self.num_layers * self.per_layer

    def head(self, layer: int, head: int) -> int:
        return layer * self.per_layer + head

    def mlp(self, layer: int) -> int:
        return layer * self.per_layer + self.num_heads

    def layer_of(self, idx) -> np.ndarray:
        return np.asarray(idx) // self.per_layer

    def is_mlp(self, idx) -> np.ndarray:
        return (np.asarray(idx) % self.per_layer) == self.num_heads

    def label(self, idx: int) -> str:
        l, r = divmod(int(idx), self.per_layer)
        return f"L{l}MLP" if r == self.num_heads else f"L{l}H{r}"

    def labels(self, idxs: Sequence[int]) -> List[str]:
        return [self.label(i) for i in idxs]

    def heads_in_layer(self, layer: int) -> np.ndarray:
        return np.arange(self.num_heads) + layer * self.per_layer


@dataclass
class Intervention:
    """One forward pass worth of edits at the final position.

    patch: component indices whose activation is replaced by the source run's
           activation (oracle activation patching).
    steer: component indices to which alpha * delta[c] is added (no oracle).
    resid_layers / resid_alpha / resid_src: residual-stream (layer-level)
           steering. resid_alpha * steer_resid[resid_src, l] is added to the
           output of decoder layer l (final position) for every l in resid_layers.
    """

    patch: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    steer: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    alpha: float = 0.0
    resid_layers: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int64)
    )
    resid_alpha: float = 0.0
    resid_src: int = 0


@dataclass
class RunState:
    """Everything needed to intervene on one prompt at its final position."""

    prefix_kv: List[Tuple[torch.Tensor, torch.Tensor]]  # per layer (K, V), batch 1
    last_token: int
    prefix_len: int
    cand_ids: torch.Tensor  # [n_cand] long (device)
    cand_logits: torch.Tensor  # [n_cand] fp32 (cpu)
    vocab_entropy: float
    vocab_stats: (
        np.ndarray
    )  # [entropy, top-1 prob, top1-top2 logit margin] over the FULL vocabulary
    z: torch.Tensor  # [L, H*d] o_proj input at last position (device, model dtype)
    mlp: torch.Tensor  # [L, D] mlp output at last position
    resid: torch.Tensor  # [L, D] decoder-layer output at last position


def vocab_stats(logits: torch.Tensor) -> np.ndarray:
    """[entropy, top-1 probability, top1 - top2 logit margin] of the next-token
    distribution over the full vocabulary (needs no candidate answers)."""
    logits = logits.float()
    logp = torch.log_softmax(logits, dim=-1)
    ent = float(-(logp.exp() * logp).sum().item())
    top = torch.topk(logits, 2).values
    return np.array(
        [ent, float(logp.max().exp().item()), float((top[0] - top[1]).item())],
        dtype=np.float64,
    )


# =====================================================================
# Engine
# =====================================================================


class CausalEngine:
    def __init__(
        self,
        model_name: str,
        dtype: str = "bfloat16",
        trust_remote_code: bool = False,
        device: Optional[str] = None,
        attn_implementation: Optional[str] = None,
    ):
        self.model_name = model_name
        log(f"[Engine] Loading tokenizer {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[dtype]
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        import transformers
        from packaging import version

        dtype_kw = (
            "dtype"
            if version.parse(transformers.__version__) >= version.parse("4.56")
            else "torch_dtype"
        )
        kwargs = {dtype_kw: torch_dtype, "trust_remote_code": trust_remote_code}
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        log(f"[Engine] Loading model ({dtype}) on {self.device}")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs).to(
            self.device
        )
        self.model.eval()
        self.dtype = torch_dtype

        self.layers = self._find_layers()
        cfg = getattr(self.model.config, "text_config", self.model.config)
        self.num_layers = len(self.layers)
        self.num_heads = int(cfg.num_attention_heads)
        self.hidden_size = int(cfg.hidden_size)
        head_dim = getattr(cfg, "head_dim", None)
        self.head_dim = (
            int(head_dim) if head_dim else self.hidden_size // self.num_heads
        )

        self.o_projs, self.mlps = [], []
        for i, layer in enumerate(self.layers):
            attn = getattr(layer, "self_attn", None)
            o_proj = getattr(attn, "o_proj", None) if attn is not None else None
            if o_proj is None:
                raise AttributeError(
                    f"Layer {i} has no self_attn.o_proj; head patching unsupported."
                )
            if o_proj.in_features != self.num_heads * self.head_dim:
                raise ValueError(
                    f"Layer {i}: o_proj.in_features={o_proj.in_features} != "
                    f"num_heads*head_dim={self.num_heads}*{self.head_dim}"
                )
            self.o_projs.append(o_proj)
            self.mlps.append(layer.mlp)

        self.space = ComponentSpace(self.num_layers, self.num_heads)
        self.has_chat_template = bool(getattr(self.tokenizer, "chat_template", None))
        log(
            f"[Engine] layers={self.num_layers} heads={self.num_heads} head_dim={self.head_dim} "
            f"hidden={self.hidden_size} components={self.space.size} chat_template={self.has_chat_template}"
        )

    # ------------------------------------------------------------------
    def _find_layers(self) -> torch.nn.ModuleList:
        for path in [
            ("model", "layers"),
            ("model", "language_model", "layers"),
            ("model", "language_model", "model", "layers"),
            ("language_model", "model", "layers"),
            ("transformer", "h"),
        ]:
            cur = self.model
            for p in path:
                cur = getattr(cur, p, None)
                if cur is None:
                    break
            if (
                isinstance(cur, torch.nn.ModuleList)
                and len(cur)
                and hasattr(cur[0], "self_attn")
            ):
                return cur
        cands = [
            (len(m), m)
            for n, m in self.model.named_modules()
            if isinstance(m, torch.nn.ModuleList)
            and len(m)
            and hasattr(m[0], "self_attn")
            and "vision" not in n.lower()
        ]
        if not cands:
            raise AttributeError(
                f"Could not locate decoder layers in {self.model_name}"
            )
        return max(cands, key=lambda x: x[0])[1]

    # ------------------------------------------------------------------
    # Prompting
    # ------------------------------------------------------------------
    def render_prompt(self, user_content: str, prefill: str = "") -> str:
        """Chat-templated prompt ending inside the assistant turn with `prefill`."""
        if self.has_chat_template:
            text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": user_content}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,  # Qwen3: inserts empty <think></think>; ignored elsewhere
            )
        else:
            bos = self.tokenizer.bos_token or ""
            text = f"{bos}{user_content}\n"
        if prefill and text and not text[-1].isspace():
            prefill = " " + prefill  # e.g. Mistral's "[/INST]" is followed by a space
        return text + prefill

    def encode(self, text: str) -> List[int]:
        # The rendered template already contains BOS where the model expects it.
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def continuation_ids(
        self, prompt_text: str, prompt_ids: List[int], cont: str
    ) -> Optional[List[int]]:
        """Token ids of `cont` as the tokenizer splits it *after* prompt_text."""
        full = self.encode(prompt_text + cont)
        n = len(prompt_ids)
        if full[:n] != prompt_ids or len(full) <= n:
            return None
        return full[n:]

    # ------------------------------------------------------------------
    # Cache helpers (portable across transformers versions)
    # ------------------------------------------------------------------
    @staticmethod
    def _cache_to_list(cache) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        if isinstance(cache, (list, tuple)):
            return [(k, v) for k, v in cache]
        if hasattr(cache, "layers"):  # transformers >= 4.56 layered cache
            return [(lay.keys, lay.values) for lay in cache.layers]
        if hasattr(cache, "key_cache"):
            return list(zip(cache.key_cache, cache.value_cache))
        if hasattr(cache, "to_legacy_cache"):
            return [(k, v) for k, v in cache.to_legacy_cache()]
        raise TypeError(f"Unsupported cache type {type(cache)}")

    def _expanded_cache(self, kv: List[Tuple[torch.Tensor, torch.Tensor]], batch: int):
        from transformers import DynamicCache

        cache = DynamicCache()
        for i, (k, v) in enumerate(kv):
            cache.update(
                k.expand(batch, *k.shape[1:]), v.expand(batch, *v.shape[1:]), i
            )
        return cache

    # ------------------------------------------------------------------
    # Base pass
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def prepare(self, input_ids: Sequence[int], cand_ids: Sequence[int]) -> RunState:
        """Run the prefix once, then the last token with activation capture."""
        ids = torch.tensor([list(input_ids)], dtype=torch.long, device=self.device)
        if ids.shape[1] < 2:
            raise ValueError("Prompt must have at least 2 tokens.")
        prefix = ids[:, :-1]
        out = self.model(input_ids=prefix, use_cache=True)
        kv = self._cache_to_list(out.past_key_values)
        kv = [(k.detach(), v.detach()) for k, v in kv]
        cand = torch.tensor(list(cand_ids), dtype=torch.long, device=self.device)

        L = self.num_layers
        z = [None] * L
        mlp = [None] * L
        resid = [None] * L
        handles = []

        def zhook(l):
            def f(mod, args):
                z[l] = args[0][0, -1].detach()

            return f

        def mhook(l):
            def f(mod, inp, outp):
                o = outp[0] if isinstance(outp, tuple) else outp
                mlp[l] = o[0, -1].detach()

            return f

        def rhook(l):
            def f(mod, inp, outp):
                o = outp[0] if isinstance(outp, tuple) else outp
                resid[l] = o[0, -1].detach()

            return f

        for l in range(L):
            handles.append(self.o_projs[l].register_forward_pre_hook(zhook(l)))
            handles.append(self.mlps[l].register_forward_hook(mhook(l)))
            handles.append(self.layers[l].register_forward_hook(rhook(l)))
        try:
            S = ids.shape[1]
            o2 = self.model(
                input_ids=ids[:, -1:],
                past_key_values=self._expanded_cache(kv, 1),
                attention_mask=torch.ones(1, S, dtype=torch.long, device=self.device),
                cache_position=torch.tensor([S - 1], device=self.device),
                use_cache=True,
            )
        finally:
            for h in handles:
                h.remove()
        logits = o2.logits[0, -1].float()
        vstats = vocab_stats(logits)
        entropy = float(vstats[0])
        return RunState(
            prefix_kv=kv,
            last_token=int(ids[0, -1].item()),
            prefix_len=int(prefix.shape[1]),
            cand_ids=cand,
            cand_logits=logits[cand].cpu(),
            vocab_entropy=entropy,
            vocab_stats=vstats,
            z=torch.stack(z),
            mlp=torch.stack(mlp),
            resid=torch.stack(resid),
        )

    @torch.inference_mode()
    def prompt_features(
        self, input_ids: Sequence[int]
    ) -> Tuple[np.ndarray, torch.Tensor]:
        """Candidate-free features at the last prompt token: full-vocabulary
        [entropy, top-1 prob, top1-top2 margin] and the per-layer residual."""
        ids = torch.tensor([list(input_ids)], dtype=torch.long, device=self.device)
        resid = [None] * self.num_layers
        handles = []
        for l in range(self.num_layers):

            def f(mod, inp, outp, l=l):
                o = outp[0] if isinstance(outp, tuple) else outp
                resid[l] = o[0, -1].detach()

            handles.append(self.layers[l].register_forward_hook(f))
        try:
            logits = self.model(input_ids=ids).logits[0, -1].float()
        finally:
            for h in handles:
                h.remove()
        return vocab_stats(logits), torch.stack(resid)

    # ------------------------------------------------------------------
    # Batched interventions
    # ------------------------------------------------------------------
    def _build_masks(self, ivs: Sequence[Intervention]):
        B, L, H = len(ivs), self.num_layers, self.num_heads
        P = np.zeros((B, self.space.size), dtype=bool)
        S = np.zeros((B, self.space.size), dtype=np.float32)
        for b, iv in enumerate(ivs):
            if len(iv.patch):
                P[b, iv.patch] = True
            if len(iv.steer) and iv.alpha != 0:
                S[b, iv.steer] += iv.alpha
        P = P.reshape(B, L, H + 1)
        S = S.reshape(B, L, H + 1)
        R = np.zeros((B, L), dtype=np.float32)
        Rsrc = np.zeros(B, dtype=np.int64)
        for b, iv in enumerate(ivs):
            if len(iv.resid_layers) and iv.resid_alpha != 0:
                R[b, np.asarray(iv.resid_layers, dtype=np.int64)] += iv.resid_alpha
                Rsrc[b] = int(iv.resid_src)
        return P, S, R, Rsrc

    def run_interventions(
        self,
        state: RunState,
        interventions: Sequence[Intervention],
        source: Optional[RunState] = None,
        steer_z: Optional[torch.Tensor] = None,
        steer_mlp: Optional[torch.Tensor] = None,
        batch_size: int = 64,
        steer_resid: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """Returns float32 candidate logits [n_interventions, n_cand].

        source:      run whose last-position activations are patched in.
        steer_z:     [L, H*d] additive head-output directions (fp32 ok).
        steer_mlp:   [L, D] additive MLP-output directions.
        steer_resid: [V, L, D] (or [L, D]) residual-stream directions; an
                     Intervention selects one of the V sets with resid_src.
        Halves the batch size and retries on CUDA OOM.
        """
        if steer_resid is not None and steer_resid.dim() == 2:
            steer_resid = steer_resid.unsqueeze(0)
        if steer_resid is not None:
            steer_resid = steer_resid.to(device=self.device, dtype=torch.float32)
        bs = min(batch_size, getattr(self, "_safe_batch", batch_size))
        while True:
            try:
                return self._run_interventions(
                    state, interventions, source, steer_z, steer_mlp, bs, steer_resid
                )
            except torch.cuda.OutOfMemoryError:
                if bs == 1:
                    raise
                torch.cuda.empty_cache()
                bs = max(1, bs // 2)
                self._safe_batch = bs
                log(f"[Engine] CUDA OOM -> batch size {bs}")

    @torch.inference_mode()
    def _run_interventions(
        self,
        state,
        interventions,
        source,
        steer_z,
        steer_mlp,
        batch_size,
        steer_resid=None,
    ):
        results = []
        H, d = self.num_heads, self.head_dim
        for start in range(0, len(interventions), batch_size):
            chunk = interventions[start : start + batch_size]
            B = len(chunk)
            P, Sc, R, Rsrc = self._build_masks(chunk)
            needs_patch = P.any(axis=0)  # [L, H+1]
            needs_steer = (Sc != 0).any(axis=0)
            needs_resid = (R != 0).any(axis=0)  # [L]
            if needs_resid.any() and steer_resid is None:
                raise ValueError("Residual steering needs steer_resid vectors.")
            if needs_patch.any() and source is None:
                raise ValueError("Patch interventions need a source RunState.")
            if needs_steer.any() and (steer_z is None or steer_mlp is None):
                raise ValueError("Steer interventions need steering vectors.")
            Pt = torch.from_numpy(P).to(self.device)
            St = torch.from_numpy(Sc).to(self.device)
            Rt = torch.from_numpy(R).to(self.device)
            Rs = torch.from_numpy(Rsrc).to(self.device)
            handles = []

            def rhook(l):
                coef = Rt[:, l]

                def f(mod, inp, outp):
                    is_t = isinstance(outp, tuple)
                    o = (outp[0] if is_t else outp).clone()
                    vec = steer_resid[Rs, l]  # [B, D]
                    o[:, -1] = (o[:, -1].float() + coef[:, None] * vec).to(o.dtype)
                    return (o,) + tuple(outp[1:]) if is_t else o

                return f

            def zhook(l):
                ph = Pt[:, l, :H]
                sh = St[:, l, :H]
                do_p = bool(needs_patch[l, :H].any())
                do_s = bool(needs_steer[l, :H].any())

                def f(mod, args):
                    x = args[0]
                    Bx, Sx, _ = x.shape
                    x = x.clone().view(Bx, Sx, H, d)
                    last = x[:, -1]  # [B, H, d]
                    if do_p:
                        src = source.z[l].view(1, H, d).to(last.dtype)
                        last = torch.where(ph[:, :, None], src, last)
                    if do_s:
                        dz = (
                            steer_z[l]
                            .view(1, H, d)
                            .to(device=last.device, dtype=torch.float32)
                        )
                        last = (last.float() + sh[:, :, None] * dz).to(x.dtype)
                    x[:, -1] = last
                    return (x.view(Bx, Sx, H * d),) + tuple(args[1:])

                return f

            def mhook(l):
                pm = Pt[:, l, H]
                sm = St[:, l, H]
                do_p = bool(needs_patch[l, H])
                do_s = bool(needs_steer[l, H])

                def f(mod, inp, outp):
                    is_t = isinstance(outp, tuple)
                    o = (outp[0] if is_t else outp).clone()
                    last = o[:, -1]
                    if do_p:
                        last = torch.where(
                            pm[:, None], source.mlp[l][None].to(last.dtype), last
                        )
                    if do_s:
                        dm = steer_mlp[l][None].to(
                            device=last.device, dtype=torch.float32
                        )
                        last = (last.float() + sm[:, None] * dm).to(o.dtype)
                    o[:, -1] = last
                    return (o,) + tuple(outp[1:]) if is_t else o

                return f

            for l in range(self.num_layers):
                if needs_patch[l, :H].any() or needs_steer[l, :H].any():
                    handles.append(self.o_projs[l].register_forward_pre_hook(zhook(l)))
                if needs_patch[l, H] or needs_steer[l, H]:
                    handles.append(self.mlps[l].register_forward_hook(mhook(l)))
                if needs_resid[l]:
                    handles.append(self.layers[l].register_forward_hook(rhook(l)))
            try:
                Sfull = state.prefix_len + 1
                out = self.model(
                    input_ids=torch.full(
                        (B, 1), state.last_token, dtype=torch.long, device=self.device
                    ),
                    past_key_values=self._expanded_cache(state.prefix_kv, B),
                    attention_mask=torch.ones(
                        B, Sfull, dtype=torch.long, device=self.device
                    ),
                    cache_position=torch.tensor([state.prefix_len], device=self.device),
                    use_cache=True,
                )
            finally:
                for h in handles:
                    h.remove()
            results.append(out.logits[:, -1].float()[:, state.cand_ids].cpu().numpy())
        if not results:
            return np.zeros((0, len(state.cand_ids)), dtype=np.float32)
        return np.concatenate(results, axis=0)

    # ------------------------------------------------------------------
    # Full-sequence reference (used only for verification)
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def _full_forward_with_last_patch(
        self, input_ids, cand_ids, source=None, patch=(), resid=None
    ):
        ids = torch.tensor([list(input_ids)], dtype=torch.long, device=self.device)
        H, d = self.num_heads, self.head_dim
        handles = []
        if resid is not None:  # (layer, alpha, vec[L, D])
            rl, ra, rv = resid

            def rh(mod, inp, outp):
                is_t = isinstance(outp, tuple)
                o = (outp[0] if is_t else outp).clone()
                o[:, -1] = (o[:, -1].float() + ra * rv[rl].to(o.device).float()).to(
                    o.dtype
                )
                return (o,) + tuple(outp[1:]) if is_t else o

            handles.append(self.layers[rl].register_forward_hook(rh))
        patch = list(patch)
        for c in patch:
            l = int(self.space.layer_of(c))
            if self.space.is_mlp(c):

                def f(mod, inp, outp, l=l):
                    o = (outp[0] if isinstance(outp, tuple) else outp).clone()
                    o[:, -1] = source.mlp[l].to(o.dtype)
                    return (o,) + tuple(outp[1:]) if isinstance(outp, tuple) else o

                handles.append(self.mlps[l].register_forward_hook(f))
            else:
                h = int(c % self.space.per_layer)

                def g(mod, args, l=l, h=h):
                    x = args[0].clone()
                    x[:, -1, h * d : (h + 1) * d] = source.z[l, h * d : (h + 1) * d].to(
                        x.dtype
                    )
                    return (x,) + tuple(args[1:])

                handles.append(self.o_projs[l].register_forward_pre_hook(g))
        try:
            logits = self.model(input_ids=ids).logits[0, -1].float()
        finally:
            for h in handles:
                h.remove()
        return logits[torch.tensor(list(cand_ids), device=self.device)].cpu().numpy()

    def verify_cached_equivalence(
        self, corrupt_ids, clean_ids, cand_ids, tol: float = 0.5
    ) -> Dict[str, float]:
        """Cached single-token interventions must match full-sequence forwards."""
        sc = self.prepare(corrupt_ids, cand_ids)
        so = self.prepare(clean_ids, cand_ids)
        sp = self.space
        mid = self.num_layers // 2
        comps = [
            sp.head(0, 0),
            sp.head(mid, self.num_heads - 1),
            sp.mlp(mid),
            sp.mlp(self.num_layers - 1),
        ]
        cached = self.run_interventions(
            sc, [Intervention(), Intervention(patch=np.array(comps))], source=so
        )
        ref_base = self._full_forward_with_last_patch(corrupt_ids, cand_ids)
        ref_patch = self._full_forward_with_last_patch(
            corrupt_ids, cand_ids, source=so, patch=comps
        )
        rv = (so.resid.float() - sc.resid.float()).cpu()
        cached_r = self.run_interventions(
            sc,
            [Intervention(resid_layers=np.array([mid]), resid_alpha=1.0)],
            steer_resid=rv,
        )
        ref_r = self._full_forward_with_last_patch(
            corrupt_ids, cand_ids, resid=(mid, 1.0, rv)
        )
        res = {
            "base_max_abs_diff": float(np.abs(cached[0] - ref_base).max()),
            "patch_max_abs_diff": float(np.abs(cached[1] - ref_patch).max()),
            "resid_steer_max_abs_diff": float(np.abs(cached_r[0] - ref_r).max()),
            "state_vs_full_max_abs_diff": float(
                np.abs(sc.cand_logits.numpy() - ref_base).max()
            ),
            "patch_effect_size": float(np.abs(ref_patch - ref_base).max()),
            "resid_steer_effect_size": float(np.abs(ref_r - ref_base).max()),
        }
        worst = max(
            res["base_max_abs_diff"],
            res["patch_max_abs_diff"],
            res["resid_steer_max_abs_diff"],
            res["state_vs_full_max_abs_diff"],
        )
        log(f"[Verify] cached-vs-full: {res}")
        if worst > 10 * tol:
            raise RuntimeError(
                f"Cached intervention path disagrees with full forward: {res}"
            )
        if worst > tol:
            warnings.warn(
                f"Cached path deviates by {worst:.3f} logits (numerical; tol={tol})."
            )
        return res

    # ------------------------------------------------------------------
    # Generation with (optional) steering, for behavioural metrics
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def generate(
        self,
        input_ids: Sequence[int],
        max_new_tokens: int = 32,
        steer: Optional[np.ndarray] = None,
        alpha: float = 0.0,
        steer_z: Optional[torch.Tensor] = None,
        steer_mlp: Optional[torch.Tensor] = None,
        resid_layers: Optional[Sequence[int]] = None,
        resid_alpha: float = 0.0,
        resid_vec: Optional[torch.Tensor] = None,
    ) -> str:
        """Greedy decoding; steering (component and/or residual-layer, if any)
        is added at the last prompt position and at every generated position."""
        H, d = self.num_heads, self.head_dim
        handles = []
        if resid_layers is not None and len(resid_layers) and resid_alpha != 0:
            rv = resid_vec.to(device=self.device, dtype=torch.float32)
            for rl in resid_layers:

                def rh(mod, inp, outp, rl=int(rl)):
                    is_t = isinstance(outp, tuple)
                    o = (outp[0] if is_t else outp).clone()
                    o[:, -1] = (o[:, -1].float() + resid_alpha * rv[rl]).to(o.dtype)
                    return (o,) + tuple(outp[1:]) if is_t else o

                handles.append(self.layers[int(rl)].register_forward_hook(rh))
        if steer is not None and len(steer) and alpha != 0:
            comps = np.asarray(steer)
            for l in sorted(set(self.space.layer_of(comps).tolist())):
                in_l = comps[self.space.layer_of(comps) == l]
                heads = [
                    int(c % self.space.per_layer)
                    for c in in_l
                    if not self.space.is_mlp(c)
                ]
                has_mlp = any(self.space.is_mlp(c) for c in in_l)
                if heads:
                    mask = torch.zeros(H, device=self.device)
                    mask[heads] = alpha

                    def g(mod, args, l=l, mask=mask):
                        x = args[0].clone()
                        Bx, Sx, _ = x.shape
                        v = x.view(Bx, Sx, H, d)
                        dz = (
                            steer_z[l]
                            .view(H, d)
                            .to(device=x.device, dtype=torch.float32)
                        )
                        v[:, -1] = (v[:, -1].float() + mask[:, None] * dz).to(x.dtype)
                        return (v.view(Bx, Sx, H * d),) + tuple(args[1:])

                    handles.append(self.o_projs[l].register_forward_pre_hook(g))
                if has_mlp:

                    def f(mod, inp, outp, l=l):
                        is_t = isinstance(outp, tuple)
                        o = (outp[0] if is_t else outp).clone()
                        dm = steer_mlp[l].to(device=o.device, dtype=torch.float32)
                        o[:, -1] = (o[:, -1].float() + alpha * dm).to(o.dtype)
                        return (o,) + tuple(outp[1:]) if is_t else o

                    handles.append(self.mlps[l].register_forward_hook(f))
        try:
            ids = torch.tensor([list(input_ids)], dtype=torch.long, device=self.device)
            out = self.model(input_ids=ids, use_cache=True)
            past = out.past_key_values
            generated: List[int] = []
            nxt = int(out.logits[0, -1].float().argmax().item())
            eos = set()
            for tok in [
                self.tokenizer.eos_token_id,
                getattr(self.model.generation_config, "eos_token_id", None),
            ]:
                if isinstance(tok, int):
                    eos.add(tok)
                elif isinstance(tok, (list, tuple)):
                    eos.update(int(t) for t in tok)
            pos = ids.shape[1]
            for _ in range(max_new_tokens):
                if nxt in eos:
                    break
                generated.append(nxt)
                out = self.model(
                    input_ids=torch.tensor([[nxt]], device=self.device),
                    past_key_values=past,
                    attention_mask=torch.ones(
                        1, pos + 1, dtype=torch.long, device=self.device
                    ),
                    cache_position=torch.tensor([pos], device=self.device),
                    use_cache=True,
                )
                past = out.past_key_values
                pos += 1
                nxt = int(out.logits[0, -1].float().argmax().item())
        finally:
            for h in handles:
                h.remove()
        return self.tokenizer.decode(generated, skip_special_tokens=True)


# =====================================================================
# Representation similarity (paired only)
# =====================================================================


def _center_gram(K: torch.Tensor) -> torch.Tensor:
    n = K.shape[0]
    H = torch.eye(n, dtype=K.dtype) - 1.0 / n
    return H @ K @ H


def _unbiased_hsic(K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """Song et al. (2012) unbiased HSIC estimator; needs n >= 4."""
    n = K.shape[0]
    K = K.clone()
    L = L.clone()
    K.fill_diagonal_(0)
    L.fill_diagonal_(0)
    ones = torch.ones(n, 1, dtype=K.dtype)
    t1 = (K * L).sum()
    t2 = (ones.T @ K @ ones) * (ones.T @ L @ ones) / ((n - 1) * (n - 2))
    t3 = 2.0 / (n - 2) * (ones.T @ K @ L @ ones)
    return ((t1 + t2 - t3) / (n * (n - 3))).squeeze()


def paired_linear_cka(X: torch.Tensor, Y: torch.Tensor, debiased: bool = True) -> float:
    """Linear CKA between two representations of the SAME n inputs (row i of X
    and row i of Y come from the same sample).  Unpaired rows are meaningless."""
    if X.shape[0] != Y.shape[0]:
        raise ValueError("paired_linear_cka needs the same number of (paired) rows.")
    X = X.double()
    Y = Y.double()
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    K = X @ X.T
    L = Y @ Y.T
    n = X.shape[0]
    if debiased and n >= 4:
        xy, xx, yy = _unbiased_hsic(K, L), _unbiased_hsic(K, K), _unbiased_hsic(L, L)
    else:
        Kc, Lc = _center_gram(K), _center_gram(L)
        xy, xx, yy = (Kc * Lc).sum(), (Kc * Kc).sum(), (Lc * Lc).sum()
    denom = torch.sqrt(torch.clamp(xx * yy, min=0))
    if denom.item() <= 1e-12:
        return float("nan")
    return float((xy / denom).item())


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.double().flatten()
    b = b.double().flatten()
    na, nb = a.norm(), b.norm()
    if na.item() == 0 or nb.item() == 0:
        return float("nan")
    return float((a @ b / (na * nb)).item())
