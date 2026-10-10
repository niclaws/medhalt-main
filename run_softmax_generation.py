#!/usr/bin/env python3
"""New standalone free-generation runner with probability-weighted steering.

Example:
 python run_softmax_generation.py \
   --results_dir /path/to/results_v4 --model med42_8b --seed 0 \
   --protocol source:FCT --selection_target FCT \
   --prompt 'Which vitamin deficiency causes scurvy?' \
   --temperature 1.5 --max_new_tokens 64

The gate outputs probabilities; the actual residual steering is multiplied by
P(on), and the model generates with that continuous effective alpha. Existing
files are imported but NOT changed. Original discovery checkpoints required.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch

import eval_downstream_all_models as original
from patching_engine import CausalEngine,log
from softmax_gate_adapter import SoftmaxGateConfig,gate_from_router
from softmax_eval_helpers import checkpoint_split,temperature_from_validation
from experiment_runner import router_features


def main():
    a=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    a.add_argument('--results_dir',required=True)
    a.add_argument('--model',choices=sorted(original.MODELS),default='med42_8b')
    a.add_argument('--seed',type=int,default=0)
    a.add_argument('--protocol',choices=['loto','pooled','source:FCT','source:NOTA'],default='source:FCT')
    a.add_argument('--selection_target',choices=['FCT','NOTA'],default='FCT')
    group=a.add_mutually_exclusive_group(required=True)
    group.add_argument('--prompt',help='One medical question or instruction')
    group.add_argument('--prompt_file',help='UTF-8 text file with one prompt')
    a.add_argument('--prefill',default='',help='Optional assistant text prefix, e.g. Answer:')
    a.add_argument('--temperature',type=float,default=1.,help='Softmax gate temperature (not sampling temperature)')
    a.add_argument('--auto_temperature',action='store_true')
    a.add_argument('--center',type=float,default=.5)
    a.add_argument('--center_mode',choices=['fixed','original_tau'],default='fixed')
    a.add_argument('--max_new_tokens',type=int,default=64)
    a.add_argument('--dtype',choices=['bfloat16','float16','float32'],default='bfloat16')
    a.add_argument('--attn_implementation',default=None,choices=['sdpa','eager','flash_attention_2'])
    a.add_argument('--compare_base',action='store_true',help='Also generate unsteered text for comparison')
    a.add_argument('--json_output',default=None,help='Optional output JSON filepath')
    args=a.parse_args()
    prompt=Path(args.prompt_file).read_text(encoding='utf8').strip() if args.prompt_file else args.prompt
    if not prompt or not prompt.strip():a.error('prompt must not be empty')
    if args.max_new_tokens<1:a.error('max_new_tokens must be >= 1')
    SoftmaxGateConfig(temperature=args.temperature,center=args.center)
    run_dir=Path(args.results_dir)/f'{args.model}_seed{args.seed}'
    if not run_dir.is_dir():raise FileNotFoundError(run_dir)
    selection=original.get_selection(run_dir,args.protocol,args.selection_target)
    vector,router=original.fit_from_discovery(run_dir,selection)
    if selection['selected'] and args.auto_temperature:
        features,wrong=[],[]
        for task in selection['selection_tasks']:
            for _,b in checkpoint_split(run_dir,task,'validation'):
                for rt in ('clean','corrupt'):
                    features.append(router_features(b,rt))
                    wrong.append(not bool(b[f'correct_{rt}']))
        temperature=temperature_from_validation(router,features,wrong)
    else:temperature=args.temperature
    center=args.center
    if selection['selected'] and args.center_mode=='original_tau':
        center=float(selection['tau'])
        if not (np.isfinite(center) and 0<center<1):
            raise ValueError(f'original tau={center} is not a valid soft center; choose fixed')
    gate_cfg=SoftmaxGateConfig(temperature=temperature,center=center)
    engine=CausalEngine(original.MODELS[args.model],dtype=args.dtype,attn_implementation=args.attn_implementation)
    rendered=engine.render_prompt(prompt,args.prefill)
    tokens=engine.encode(rendered)
    if selection['selected']:
        stats,resid=engine.prompt_features(tokens)
        # Vocab-only router never receives gold answers or candidate logits.
        feat={'vocab_stats':np.asarray(stats),'resid':resid.float().cpu()}
        gate=gate_from_router(router,feat,float(selection['alpha']),gate_cfg)
        alpha_eff=gate.alpha_effective
        opts={'resid_layers':[int(selection['layer'])],
              'resid_alpha':alpha_eff,'resid_vec':vector}
        risk=gate.risk;off=gate.p_off;on=gate.p_on
    else:
        alpha_eff=0.;opts={};risk=None;off=1.;on=0.
    text=engine.generate(tokens,max_new_tokens=args.max_new_tokens,**opts)
    baseline=engine.generate(tokens,max_new_tokens=args.max_new_tokens) if args.compare_base else None
    row={
        'model':args.model,'seed':args.seed,'protocol':args.protocol,
        'selection_target':args.selection_target,'selected':bool(selection['selected']),
        'risk_score':risk,'gate_off_prob':off,'gate_on_prob':on,
        'gate_temperature':temperature,'gate_center':center,
        'alpha_base':float(selection['alpha']) if selection['selected'] else 0.,
        'alpha_effective':alpha_eff,
        'layer':int(selection['layer']) if selection['selected'] else None,
        'generated_text_soft':text,'generated_text_base':baseline,
        'note':'Gate probabilities are not calibrated probabilities of clinical factuality.'
    }
    output=json.dumps(row,indent=2,ensure_ascii=False)
    print(output)
    if args.json_output:
        dst=Path(args.json_output);dst.parent.mkdir(parents=True,exist_ok=True)
        dst.write_text(output+'\n',encoding='utf8');log(f'Saved {dst}')

if __name__=='__main__':main()
