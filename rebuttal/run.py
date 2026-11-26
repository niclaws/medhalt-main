"""Entry point. Run `python -m rebuttal.run --help` from the existing repository."""
from __future__ import annotations
import argparse, copy, gc, hashlib, importlib.metadata, json, logging, os, platform, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .core import MODELS,LARGE_MODELS,TASKS,ALPHAS,RHOS,Store,aggregate,write_json
from .checkpoints import Run
from ._timing import LOG, timed, timed_call, step, configure_logging, LOG_LEVELS

EXPS=('audit','format','attenuation','strength','treatment','probe','scale')
DEFAULT_MODELS=[m for m in MODELS if m not in LARGE_MODELS]

def parser():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--experiments',nargs='+',choices=EXPS,default=['audit','format','attenuation','strength','treatment','probe'])
    p.add_argument('--models',nargs='+',choices=sorted(MODELS),default=DEFAULT_MODELS); p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2])
    p.add_argument('--results-dir',default='results/results_v4'); p.add_argument('--output-dir',default='results/rebuttal'); p.add_argument('--data-dir',default='medhalt/datasets')
    p.add_argument('--splits',nargs='+',choices=['test','unfiltered'],default=['test','unfiltered']); p.add_argument('--sources',nargs='+',choices=TASKS,default=list(TASKS)); p.add_argument('--targets',nargs='+',choices=TASKS,default=list(TASKS)); p.add_argument('--limit',type=int,default=None,help='Explicit pilot cap per saved evaluation split; not used for external evaluation'); p.add_argument('--train-limit',type=int,default=None,help='Optional discovery-pair cap for treatment router only')
    p.add_argument('--batch-size',type=int,default=16); p.add_argument('--dtype',choices=['bfloat16','float16','float32'],default=None,help='Original checkpoint dtype for small models unless explicitly overridden (mismatches refused)'); p.add_argument('--attn-implementation',choices=['sdpa','eager','flash_attention_2'],default=None); p.add_argument('--engine',choices=['auto','legacy','sharded'],default='auto'); p.add_argument('--trust-remote-code',action='store_true'); p.add_argument('--revision',default=None); p.add_argument('--max-memory',default=None,help='JSON mapping e.g. {"0":"70GiB","1":"70GiB","2":"70GiB"} for sharded loader'); p.add_argument('--allow-cpu-offload',action='store_true')
    p.add_argument('--quantization',choices=['auto','none','nf4'],default='auto',help='Auto: NF4 for 27B/70B, unchanged precision for existing models. NF4 requires exactly one visible GPU.')
    p.add_argument('--k',type=int,default=20); p.add_argument('--alphas',nargs='+',type=float,default=list(ALPHAS)); p.add_argument('--rhos',nargs='+',type=float,default=list(RHOS)); p.add_argument('--random-reps',type=int,default=3); p.add_argument('--shuffle-reps',type=int,default=5); p.add_argument('--gate-caps',nargs='*',type=float,default=[],help='Optional capped soft-gate controls, e.g. 2 1 .5 .25; uncapped always included'); p.add_argument('--temperature',type=float,default=1.5); p.add_argument('--center',type=float,default=.5); p.add_argument('--harm-eps',type=float,default=.10); p.add_argument('--probe-c',type=float,default=1.)
    p.add_argument('--n-medqa',type=int,default=200); p.add_argument('--n-pubmedqa',type=int,default=200); p.add_argument('--sample-seed',type=int,default=42); p.add_argument('--treatment-external',action='store_true'); p.add_argument('--treatment-extra-pairs',type=int,default=0,help='Optional fresh source-only outcome-label cohort, disjoint from all original FCT/NOTA pairs'); p.add_argument('--reference-soft-dir',default=None,help='Optional original softmax output root: verify exact accepted IDs/golds and log drift'); p.add_argument('--external-samples-json',default=None,help='Optional frozen manifest from previous rebuttal run; no datasets redownload')
    p.add_argument('--large-n-samples',type=int,default=1000); p.add_argument('--large-max-pairs',type=int,default=800); p.add_argument('--large-discovery-limit',type=int,default=32,help='Declared scale pilot discovery cap; 0 means all discovery pairs'); p.add_argument('--large-mode',choices=['layer','component'],default='layer'); p.add_argument('--large-all-layers',action='store_true'); p.add_argument('--large-random-controls',type=int,default=5)
    p.add_argument('--log-level', choices=LOG_LEVELS, type=str.upper, default='DEBUG', help='Console and file verbosity; timestamps always IST')
    p.add_argument('--resume',action='store_true'); p.add_argument('--fail-fast',action='store_true'); p.add_argument('--plan',action='store_true',help='Print checkpoint paths/experiment plan; no model load'); p.add_argument('--aggregate-only',action='store_true'); return p

def settings(args):
    x=vars(args).copy()
    for k in ('resume','fail_fast','plan','aggregate_only','models','seeds','experiments','output_dir','log_level'): x.pop(k,None)
    return x

def setup_logging(root, level='DEBUG'):
    configure_logging(Path(root)/'rebuttal.log', level)

@timed
def versions():
    libs={}
    for x in ('torch','transformers','accelerate','bitsandbytes','numpy','pandas','scipy','scikit-learn','datasets','huggingface-hub'):
        try: libs[x]=importlib.metadata.version(x)
        except importlib.metadata.PackageNotFoundError: libs[x]=None
    return dict(python=sys.version,platform=platform.platform(),libraries=libs,cuda=torch.version.cuda,cuda_available=torch.cuda.is_available(),gpu=[dict(name=torch.cuda.get_device_name(i),memory=torch.cuda.get_device_properties(i).total_memory) for i in range(torch.cuda.device_count())])

@timed
def provenance(run):
    files=[run.path/'run_config.json',run.path/'splits.json',run.path/'safe_layer'/'safe_layer_selection.csv']+list(run.state.glob('pairs_*.pt'))+list(run.state.glob('base_*.pt'))+list(run.state.glob('vectors_*.pt'))+list(run.state.glob('selection_*.pt'))
    # Large tensor checkpoints get file stat fingerprint; small configuration files full hash.
    return {str(p.relative_to(run.path)):dict(bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns,**({'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} if p.stat().st_size<2_000_000 else {})) for p in files if p.is_file()}

@timed
def code_fingerprint():
    root=Path(__file__).resolve().parent.parent; paths=list((root/'rebuttal').glob('*.py'))+[root/x for x in ('patching_engine.py','experiment_runner.py','run_casual_transfer.py','safe_layer_steering.py','softmax_gate_adapter.py','eval_downstream_all_models.py')]
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.exists()}

@timed
def engine_for(tag,args,run):
    from patching_engine import CausalEngine
    name=MODELS[tag]; cfg=run.config if run else {}; old=cfg.get('model_name')
    if old and old!=name: raise ValueError(f'Checkpoint model_name={old} differs from requested {name}')
    dtype=args.dtype or cfg.get('dtype','bfloat16'); olddtype=cfg.get('dtype')
    if olddtype and dtype!=olddtype: raise ValueError('Changing original dtype is not a faithful checkpoint reuse; use a fresh run')
    attn=args.attn_implementation or cfg.get('attn_implementation')
    from .engine import resolve_quantization
    quantization=resolve_quantization(tag,getattr(args,'quantization','auto'),reuse_checkpoint=run is not None)
    if args.engine=='sharded' or tag in LARGE_MODELS or quantization=='nf4':
        if args.engine=='legacy' and (tag in LARGE_MODELS or quantization=='nf4'): raise ValueError('Large/NF4 models require sharded engine')
        from .engine import ShardedEngine
        maxmem=json.loads(args.max_memory) if args.max_memory else None
        if maxmem: maxmem={int(k) if str(k).isdigit() else k:v for k,v in maxmem.items()}
        return ShardedEngine(name,dtype,args.trust_remote_code,attn or 'sdpa',maxmem,args.revision,args.allow_cpu_offload,quantization=quantization)
    if args.revision: raise ValueError('--revision only available for sharded engine; legacy engine preserves original loader')
    return timed_call('CausalEngine', CausalEngine, name,dtype=dtype,trust_remote_code=args.trust_remote_code or cfg.get('trust_remote_code',False),attn_implementation=attn)

@timed
def load_external(args,root):
    from eval_downstream_all_models import load_samples
    if args.external_samples_json:
        manifest=json.loads(Path(args.external_samples_json).read_text()); samples=manifest['samples'];
        if manifest['sample_seed']!=args.sample_seed or len(samples['medqa'])!=args.n_medqa or len(samples['pubmedqa'])!=args.n_pubmedqa: raise ValueError('External manifest does not match requested sample settings')
    else: samples={ds:timed_call('load_samples', load_samples, ds,n,args.sample_seed) for ds,n in [('medqa',args.n_medqa),('pubmedqa',args.n_pubmedqa)]}
    for ds,items in samples.items():
        if len(items)!=len({x['dataset_index'] for x in items}): raise ValueError(f'Duplicate external dataset IDs: {ds}')
    manifest=dict(sample_seed=args.sample_seed,samples=samples); f=root/'external_samples.json'
    if f.exists() and json.loads(f.read_text())!=manifest: raise ValueError('External data changed in output directory; use frozen manifest or new output path')
    write_json(f,manifest); return samples

@timed
def selected_policy(run,source):
    cfg=timed_call('run.selection', run.selection, 'source:'+source,source)
    if cfg['selected'] and (cfg['selection_tasks']!=[source] or cfg['vec']!=source): raise ValueError('Source-only policy mismatch')
    return cfg

@timed
def run_small(tag,args,root,samples):
    run=timed_call('Run', Run, Path(args.results_dir)/f'{tag}_seed{args.seed}'); cfg=dict(**settings(args),model=tag,original=provenance(run),code=code_fingerprint()); dest=root/f'{tag}_seed{args.seed}'; store=Store(dest,cfg,args.resume)
    write_json(dest/'environment.json',versions()); write_json(dest/'original_run_config.json',run.config)
    if 'audit' in args.experiments:
        def audit(d,emit): timed_call('run.audit', run.audit, d)
        store.condition('audit',audit,experiment='audit',model=tag,seed=args.seed)
    exps=set(args.experiments)-{'audit'}
    if not exps: return
    if 'scale' in exps: raise ValueError('Use --experiments scale only with large models')
    # Verify exact source inputs before allocating GPU memory.
    for t in TASKS: timed_call('run.load', run.load, t)
    if not torch.cuda.is_available(): raise RuntimeError('GPU unavailable. CPU-only audit/plan supported; actual inference requires CUDA.')
    engine=engine_for(tag,args,run); write_json(dest/'model_runtime.json',dict(model_name=engine.model_name,layers=engine.num_layers,heads=engine.num_heads,head_dim=engine.head_dim,hidden_size=engine.hidden_size,dtype=str(engine.dtype),device_map="auto",max_memory={
    0: "6GiB",
    1: "6GiB",
    2: "6GiB",
    3: "6GiB",
},model_commit=getattr(engine.model.config,'_commit_hash',None),tokenizer_class=type(engine.tokenizer).__name__,engine=type(engine).__name__))
    try:
        if 'attenuation' in exps:
            from .layer_controls import run_attenuation
            status=[]; selections={}
            for heldout in TASKS:
                cfg=timed_call('run.selection', run.selection, 'loto',heldout); source='+'.join([x for x in cfg['selection_tasks'] if x in TASKS]);
                selections[heldout]=(cfg,source)
                for ds,s in samples.items():
                    status.append(dict(heldout=heldout,dataset=ds,selected=cfg['selected'],layer=cfg['layer'],alpha=cfg['alpha'],source=source))
            timed_call('to_csv', pd.DataFrame(status).to_csv, dest/'attenuation_policy_status.csv',index=False)
            for heldout,(cfg,source) in selections.items():
                for ds,s in samples.items():
                    if cfg['selected']: store.condition(f'attenuation/loto_{heldout}_{ds}',lambda d,e:run_attenuation(engine,run,cfg,s,args,d,e),experiment='attenuation',model=tag,seed=args.seed,source=source,target=ds,split='external',heldout=heldout)
                    else: LOG.info('External LOTO abstention %s/%s',heldout,ds)
        if 'format' in exps:
            from .format_controls import run_format
            for source in args.sources:
                for target in args.targets:
                    for split in args.splits:
                        store.condition(f'format/{source}_{target}_{split}',lambda d,e:run_format(engine,run,source,target,split,args,d,e),experiment='format',model=tag,seed=args.seed,source=source,target=target,split=split)
        if exps&{'strength','treatment','probe'}:
            from .layer_controls import run_strength,fit_benefit,run_benefit,run_benefit_external,calibrate_probe,run_probe,treatment_diagnostics
            for source in args.sources:
                cfg=selected_policy(run,source); write_json(dest/f'source_policy_{source}.json',cfg)
                if not cfg['selected']:
                    LOG.warning('Original source policy abstained (%s); skip strength/treatment/probe for this source, do not substitute target-tuned layer',source); continue
                if 'strength' in exps:
                    for target in args.targets:
                        for split in args.splits: store.condition(f'strength/{source}_{target}_{split}',lambda d,e:run_strength(engine,run,source,target,split,cfg,args,d,e),experiment='strength',model=tag,seed=args.seed,source=source,target=target,split=split)
                if 'treatment' in exps:
                    # Train/calibrate once per source and evaluate all targets; raw JSONL retained if interrupted.
                    def treatment(d,emit):
                        fit=fit_benefit(engine,run,source,cfg,args,d)
                        for target in args.targets:
                            for split in args.splits:
                                sd=d/f'{target}_{split}'; sd.mkdir(exist_ok=True); run_benefit(engine,run,source,target,split,cfg,fit,args,sd,lambda r:emit({**r,'target':target,'split':split}))
                        if args.treatment_external:
                            for ds,s in samples.items():
                                sd=d/ds; sd.mkdir(exist_ok=True); run_benefit_external(engine,cfg,fit,s,args,sd,lambda r:emit({**r,'target':ds,'split':'external'}))
                        treatment_diagnostics(d)
                    store.condition(f'treatment/{source}',treatment,experiment='treatment',model=tag,seed=args.seed,source=source)
                if 'probe' in exps:
                    def probe(d,emit):
                        fit=calibrate_probe(engine,run,source,cfg,args,d)
                        for target in args.targets:
                            for split in args.splits:
                                sd=d/f'{target}_{split}'; sd.mkdir(exist_ok=True); run_probe(engine,run,source,target,split,cfg,fit,args,sd,lambda r:emit({**r,'target':target,'split':split}))
                    store.condition(f'probe/{source}',probe,experiment='probe',model=tag,seed=args.seed,source=source)
    finally:
        del engine; timed_call('gc.collect', gc.collect); timed_call('torch.cuda.empty_cache', torch.cuda.empty_cache)

@timed
def run_large(tag,args,root):
    if set(args.experiments)!={'scale'}: raise ValueError('Scale requires --experiments scale (separate fresh-run output)')
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    from .large import prepare_new_run,run_scale_layer,run_scale_component
    dest=root/f'{tag}_seed{args.seed}'; store=Store(dest,dict(**settings(args),model=tag,code=code_fingerprint()),args.resume); write_json(dest/'environment.json',versions())
    engine=engine_for(tag,args,None); write_json(dest/'model_runtime.json',dict(model_name=engine.model_name,model_commit=getattr(engine.model.config,'_commit_hash',None),device_map="auto",max_memory={
    0: "8GiB",
    1: "8GiB",
    2: "8GiB",
    3: "8GiB",
},layers=engine.num_layers,heads=engine.num_heads,head_dim=engine.head_dim,hidden_size=engine.hidden_size,dtype=str(engine.dtype),quantization=getattr(engine,'quantization_metadata',{'mode':'none'})))
    try:
        run=prepare_new_run(engine,args,dest/'new_checkpoints',MODELS[tag]); timed_call('run.audit', run.audit, dest/'data_audit')
        fn=run_scale_layer if args.large_mode=='layer' else run_scale_component
        for source in args.sources: store.condition(f'scale_{args.large_mode}/{source}',lambda d,e:fn(engine,run,source,args,d,e),experiment='scale_'+args.large_mode,model=tag,seed=args.seed,source=source)
    finally: del engine; timed_call('gc.collect', gc.collect); timed_call('torch.cuda.empty_cache', torch.cuda.empty_cache)

def main(argv=None):
    p=parser(); args=p.parse_args(argv); root=Path(args.output_dir).resolve(); setup_logging(root,args.log_level)
    try:
        with step('Complete experiment run', logging.INFO):
            result = _run(args, root, p)
        LOG.log(logging.ERROR if result else logging.INFO, 'Run exit status=%d', result)
        return result
    except BaseException:
        LOG.exception('Experiment entry point failed'); raise

def _run(args, root, p):
    if args.aggregate_only: aggregate(root); return 0
    for key in ('batch_size','random_reps','shuffle_reps','n_medqa','n_pubmedqa','k','large_n_samples','large_max_pairs'):
        if getattr(args,key)<=0: p.error(f'--{key.replace("_","-")} must be positive')
    if args.treatment_extra_pairs<0 or args.large_discovery_limit<0 or (args.limit is not None and args.limit<=0) or (args.train_limit is not None and args.train_limit<=0): p.error('Limits must be positive (large discovery limit may be 0)')
    if any(not np.isfinite(a) or a<0 for a in args.alphas) or 0 not in args.alphas or any(not np.isfinite(r) or r<=0 for r in args.rhos+args.gate_caps): p.error('alpha grid must contain zero, strengths finite and nonnegative; rho/caps positive')
    if not 0<args.harm_eps<1 or args.probe_c<=0: p.error('Invalid safety/probe parameters')
    if len(set(args.models))!=len(args.models) or len(set(args.seeds))!=len(args.seeds): p.error('Duplicate models/seeds')
    args.alphas=sorted(set(args.alphas)); args.rhos=sorted(set(args.rhos)); args.gate_caps=sorted(set(args.gate_caps)); args.data_dir=str(Path(args.data_dir).resolve()); args.results_dir=str(Path(args.results_dir).resolve())
    if args.plan:
        for tag in args.models:
            for seed in args.seeds: print(f'{tag} seed={seed} experiments={args.experiments} input={Path(args.results_dir)/f"{tag}_seed{seed}" if tag not in LARGE_MODELS and set(args.experiments)!={'scale'} else "fresh"} output={root/f"{tag}_seed{seed}"}')
        return 0
    LOG.info('Arguments %s',vars(args)); samples=load_external(args,root) if 'attenuation' in args.experiments or ('treatment' in args.experiments and args.treatment_external) else {}; failures=[]
    for tag in args.models:
        for seed in args.seeds:
            local=copy.copy(args); local.seed=seed; start=time.perf_counter(); LOG.info('START MODEL %s seed=%s',tag,seed)
            try:
                if tag in LARGE_MODELS or set(args.experiments)=={'scale'}: run_large(tag,local,root)
                else: run_small(tag,local,root,samples)
            except Exception as e:
                LOG.exception('Run failed %s seed=%s',tag,seed); failures.append(dict(model=tag,seed=seed,error=repr(e)))
                if args.fail_fast: write_json(root/'failures.json',failures); raise
            finally: LOG.info('END MODEL %s seed=%s elapsed=%.1fs',tag,seed,time.perf_counter()-start)
    aggregate(root); write_json(root/'failures.json',failures)
    if 'attenuation' in args.experiments:
        fs=list(root.glob('*_seed*/attenuation_policy_status.csv'))
        if fs:
            status=pd.concat([timed_call('pd.read_csv', pd.read_csv, f).assign(run=f.parent.name) for f in fs],ignore_index=True); timed_call('to_csv', status.to_csv, root/'attenuation_cohort.csv',index=False); active=int(status.selected.sum()); LOG.info('Active external conditions %s/%s; original paper expects 26/48 for all 4 models and 3 seeds',active,len(status));
            completed=sum(1 for f in root.glob('*_seed*/attenuation/*/DONE.json')); LOG.info('Completed external conditions: %s (selected: %s)',completed,active)
            if set(args.models)==set(DEFAULT_MODELS) and set(args.seeds)=={0,1,2} and (active!=26 or completed!=26): LOG.warning('Active or completed cohort differs from Table 7; do not claim exact replication')
    return int(bool(failures))

if __name__=='__main__': raise SystemExit(main())
