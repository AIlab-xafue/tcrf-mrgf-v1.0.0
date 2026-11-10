from pathlib import Path
import json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from src.models.baselines import mlp_fit,classic_fit,probability,threshold_grid,four_metrics
from src.training.train_model import read,ROOT
from src.training.matched_threshold import choose_valid_threshold

def main():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    out=ROOT/'outputs/financial_baselines'
    out.mkdir(parents=True,exist_ok=False)
    schema=json.loads((ROOT/'data/metadata/feature_schema.json').read_text())
    labels=read(ROOT/'data/metadata/labels.csv')
    node=read(ROOT/'data/model_input/node_features.csv')
    x={s:node.loc[node.split.eq(s),schema['finance_features']].to_numpy(dtype=np.float64) for s in ['Train','Valid','Test']}
    y={s:labels.loc[labels.split.eq(s),'distress_label'].to_numpy(dtype=np.int64) for s in x}
    ratio=float((y['Train']==0).sum()/y['Train'].sum())
    rows=[]
    for name in ['LR','SVM','XGBoost','MLP']:
        for seed in schema['formal_seeds']:
            if name=='MLP':model,vp,_,_=mlp_fit(x['Train'],y['Train'],x['Valid'],y['Valid'],seed,ratio)
            else:model,vp,_,_=classic_fit(name,x['Train'],y['Train'],x['Valid'],seed,ratio,{})
            threshold,_,_=threshold_grid(y['Valid'],vp)
            p=probability(model,x['Test'],name=='MLP')
            rows.append({'model':name,'seed':seed,**four_metrics(y['Test'],p,threshold)})
            if name=='LR':
                reference=pd.read_csv(ROOT/'data/metadata/full_validation_reference.csv').set_index('seed')
                matched,_,_=choose_valid_threshold(y['Valid'],vp,int(reference.loc[seed,'full_validation_tp']))
                rows.append({'model':'LR_matched','seed':seed,**four_metrics(y['Test'],p,matched)})
            pd.DataFrame(rows).to_csv(out/'per_seed_metrics.csv',index=False)

if __name__=='__main__':
    main()
