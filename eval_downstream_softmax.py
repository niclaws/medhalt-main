#!/usr/bin/env python3
"""New non-invasive probabilistic softmax-gated MedQA / PubMedQA runner.

Place this file and the two adapters beside the ORIGINAL files:
  eval_downstream_all_models.py, patching_engine.py,
  experiment_runner.py, safe_layer_steering.py

The original code is imported for checkpoint reconstruction and tokenization,
never edited. Per question, the fitted error-risk score is passed through a
2-class softmax; P(on) CONTINUOUSLY scales residual-stream alpha. A fresh model
forward evaluates soft steering. Predictions include answer probabilities, gate
probability, and effective alpha; no risk>tau Boolean gate is used.

Requires your local ORIGINAL .pt checkpoints in --results_dir/*_seed*/state.
The archival research ZIP created with --skip-checkpoints is NOT sufficient.
"""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import eval_downstream_all_models as original
from experiment_runner import router_features
from patching_engine import CausalEngine, log
from softmax_gate_adapter import (
    SoftmaxGateConfig,ProbabilisticResidualSteerer,gate_from_router,candidate_probabilities,
)
from softmax_eval_helpers import checkpoint_split,temperature_from_validation,summarize_predictions


def resolved_center(cfg: dict, args: argparse.Namespace) -> float:
    if args.center_mode == 'fixed':return float(args.center)
    tau=float(cfg['tau'])
    if not np.isfinite(tau) or not 0 < tau < 1:
        raise ValueError(
            f'Original hard-gate tau={tau} is not usable as a probabilistic center. '
            'Use --center_mode fixed (default), or recalibrate a finite center on validation.'
        )
    return tau


def auto_temperature(run_dir: Path, cfg: dict, router) -> float:
    features,wrong=[],[]
    for task in cfg['selection_tasks']:
        for _,b in checkpoint_split(run_dir,task,'validation'):
            for run_type in ('clean','corrupt'):
                features.append(router_features(b,run_type))
                wrong.append(not bool(b[f'correct_{run_type}']))
    return temperature_from_validation(router,features,wrong)


def evaluate(engine, samples, cfg, vector, router, gate_cfg, batch_size):
    rows,rejected=[],[]
    active=bool(cfg['selected'])
    steerer=ProbabilisticResidualSteerer(
        engine,vector,layer=int(cfg['layer']),alpha=float(cfg['alpha']),batch_size=batch_size,
    ) if active else None
    for idx,sample in enumerate(tqdm(samples,desc='Softmax steering',leave=False)):
        try:
            state,prompt_ids,shared_prefix=original.prepare_choices(engine,sample)
        except ValueError as exc:
            rejected.append({'sample_id':idx,'dataset_index':sample['dataset_index'],'reason':str(exc)})
            continue

        base_logits=np.asarray(state.cand_logits,dtype=np.float64)
        baseline_probs=candidate_probabilities(base_logits)
        risk=float('nan');prob_off,prob_on=(1.,0.)
        alpha_eff=0.
        always_logits=base_logits.copy()
        soft_logits=base_logits.copy()
        if active:
            # Prefix must NOT leak candidate-answer information into the router.
            if shared_prefix:
                stats,resid=engine.prompt_features(prompt_ids)
            else:
                stats,resid=state.vocab_stats,state.resid
            feature={'vocab_stats':np.asarray(stats), 'resid':resid.float().cpu(),
                     'cand_logits':base_logits}
            gate=gate_from_router(router,feature,steerer.alpha,gate_cfg)
            risk=gate.risk
            prob_off,prob_on=gate.p_off,gate.p_on
            alpha_eff=gate.alpha_effective
            soft_logits=steerer.run(state,alpha_eff)
            always_logits=steerer.run_always(state)
        soft_probs=candidate_probabilities(soft_logits)
        always_probs=candidate_probabilities(always_logits)
        gold=int(sample['answers'].index(sample['gold']))
        preds={
            'base':int(np.argmax(base_logits)),
            'soft':int(np.argmax(soft_logits)),
            'always':int(np.argmax(always_logits)),
        }
        rows.append({
            'sample_id':idx,'dataset_index':sample['dataset_index'],
            'gold':sample['gold'],'pred_base':sample['answers'][preds['base']],
            'pred_always':sample['answers'][preds['always']],
            'pred_soft':sample['answers'][preds['soft']],
            'correct_base':bool(preds['base']==gold),
            'correct_always':bool(preds['always']==gold),
            'correct_soft':bool(preds['soft']==gold),
            'risk_score':risk,'gate_off_prob':float(prob_off),
            'gate_on_prob':float(prob_on),
            'alpha_effective':float(alpha_eff),'alpha_selected':float(cfg['alpha']) if active else 0.,
            'gate_temperature':float(gate_cfg.temperature), 'gate_center':float(gate_cfg.center),
            'gold_prob_base':float(baseline_probs[gold]),
            'gold_prob_soft':float(soft_probs[gold]),
            'gold_prob_always':float(always_probs[gold]),
            'max_candidate_prob_soft':float(soft_probs.max()),
            'candidate_labels':json.dumps(sample['answers']),
            'candidate_prob_base':json.dumps(baseline_probs.tolist()),
            'candidate_prob_soft':json.dumps(soft_probs.tolist()),
            'candidate_prob_always':json.dumps(always_probs.tolist()),
        })
    if not rows:raise RuntimeError('No valid aligned candidate samples')
    return pd.DataFrame(rows),pd.DataFrame(rejected)


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--results_dir',required=True,help='Root containing model_seed run dirs with original .pt checkpoints')
    parser.add_argument('--models',nargs='+',choices=sorted(original.MODELS),default=['med42_8b'])
    parser.add_argument('--seeds',nargs='+',type=int,default=[0])
    parser.add_argument('--protocol',choices=['loto','pooled','source:FCT','source:NOTA'],default='loto')
    parser.add_argument('--selection_targets',nargs='+',choices=['FCT','NOTA'],default=['FCT','NOTA'])
    parser.add_argument('--datasets',nargs='+',choices=list(original.DATASETS),default=['medqa','pubmedqa'])
    parser.add_argument('--n_medqa',type=int,default=200)
    parser.add_argument('--n_pubmedqa',type=int,default=200)
    parser.add_argument('--sample_seed',type=int,default=42)
    parser.add_argument('--temperature',type=float,default=1.,help='2-class softmax temperature; ignored if --auto_temperature')
    parser.add_argument('--auto_temperature',action='store_true',help='Fit error-risk temperature on original Med-HALT VALIDATION only')
    parser.add_argument('--center_mode',choices=['fixed','original_tau'],default='fixed')
    parser.add_argument('--center',type=float,default=.5,help='Neutral risk probability for gate p_on=.5')
    parser.add_argument('--harm_eps',type=float,default=.10)
    parser.add_argument('--bootstrap_samples',type=int,default=2000)
    parser.add_argument('--dtype',choices=['bfloat16','float16','float32'],default='bfloat16')
    parser.add_argument('--attn_implementation',default=None,choices=['flash_attention_2','sdpa','eager'])
    parser.add_argument('--batch_size',type=int,default=64)
    parser.add_argument('--output_dir',default='./downstream_softmax_results')
    args=parser.parse_args()
    if min(args.n_medqa,args.n_pubmedqa,args.bootstrap_samples,args.batch_size)<=0:parser.error('All sample and batch counts must be positive')
    if not (0 < args.harm_eps < 1):parser.error('harm_eps must be in (0,1)')
    # Fail early on invalid fixed-config values even if a model abstains.
    SoftmaxGateConfig(temperature=args.temperature,center=args.center)
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    with (out/'softmax_config.json').open('w') as f:json.dump(vars(args),f,indent=2)
    all_samples={d:original.load_samples(d,args.n_medqa if d=='medqa' else args.n_pubmedqa,args.sample_seed) for d in args.datasets}
    results=[]
    for tag in args.models:
        for seed in args.seeds:
            run_dir=Path(args.results_dir)/f'{tag}_seed{seed}'
            if not run_dir.is_dir():
                log(f'[skip] absent run directory: {run_dir}')
                continue
            selected={}
            for target in args.selection_targets:
                try:selected[target]=original.get_selection(run_dir,args.protocol,target)
                except (FileNotFoundError,ValueError) as exc:log(f'[skip] no selection for {target}: {exc}')
            if not selected:continue
            engine=CausalEngine(original.MODELS[tag],dtype=args.dtype,attn_implementation=args.attn_implementation)
            try:
                for target,cfg in selected.items():
                    vector,router=original.fit_from_discovery(run_dir,cfg)
                    if args.auto_temperature and cfg['selected']:
                        temperature=auto_temperature(run_dir,cfg,router)
                    else:temperature=args.temperature
                    center=resolved_center(cfg,args) if cfg['selected'] else args.center
                    gate_cfg=SoftmaxGateConfig(temperature=temperature,center=center)
                    label=f'{tag}_seed{seed}_{args.protocol.replace(":","-")}_{target}'
                    for dataset_name,samples in all_samples.items():
                        predictions,rejected=evaluate(engine,samples,cfg,vector,router,gate_cfg,args.batch_size)
                        predictions.to_csv(out/f'{label}_{dataset_name}_softmax_predictions.csv',index=False)
                        rejected.to_csv(out/f'{label}_{dataset_name}_rejected.csv',index=False)
                        metric=summarize_predictions(predictions,seed=args.sample_seed+seed,reps=args.bootstrap_samples,harm_eps=args.harm_eps)
                        metric.update({'model_tag':tag,'seed':seed,'model_name':original.MODELS[tag],
                                       'protocol':args.protocol,'selection_target':target,
                                       'selection_tasks':'+'.join(cfg['selection_tasks']),'selection_vec':cfg['vec'],
                                       'selected':bool(cfg['selected']),
                                       'layer':int(cfg['layer']) if cfg['selected'] else -1,
                                       'alpha':float(cfg['alpha']) if cfg['selected'] else 0.,
                                       'original_hard_tau':float(cfg['tau']),
                                       'gate_temperature':temperature,'gate_center':center,
                                       'dataset':dataset_name,'dataset_split':original.DATASETS[dataset_name][2],
                                       'sample_seed':args.sample_seed,'n_requested':len(samples),
                                       'n_rejected':len(rejected),'harm_eps':args.harm_eps})
                        results.append(metric)
                        log(f'[softmax] {label}/{dataset_name}: baseline={metric["acc_base"]:.3f} '
                            f'always={metric["acc_always"]:.3f} soft={metric["acc_soft"]:.3f} '
                            f'mean P(on)={metric["mean_p_on"]:.3f} safety_UCB={metric["ucb_soft"]:.3f}')
            finally:
                del engine
                gc.collect()
                if torch.cuda.is_available():torch.cuda.empty_cache()
    if not results:raise RuntimeError('No evaluation completed; verify results_dir, selection rows and checkpoints')
    df=pd.DataFrame(results)
    df.to_csv(out/'softmax_downstream_summary.csv',index=False)
    df.groupby(['model_tag','selection_target','dataset'],as_index=False).agg(
        n_conditions=('acc_soft','size'),acc_base_mean=('acc_base','mean'),
        acc_always_mean=('acc_always','mean'),acc_soft_mean=('acc_soft','mean'),
        delta_soft_mean=('delta_soft','mean'),mean_p_on=('mean_p_on','mean'),
        mean_alpha_effective=('mean_alpha_effective','mean'),
        n_ucb_safe=('safety_ucb_soft_le_eps','sum')
    ).to_csv(out/'softmax_downstream_over_seeds.csv',index=False)
    log(f'Wrote {len(df)} evaluation summaries to {out}')


if __name__=='__main__':main()
