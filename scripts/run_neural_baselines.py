from pathlib import Path
import copy,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from src.training.train_model import ROOT,read
from src.training.training_utils import seed_all,metrics
from src.training.sequence_selection import selector
from src.models.sequence_baselines import CNN,RNN
from src.models.structural_baseline import Net

def main():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    out=ROOT/'outputs/neural_baselines';out.mkdir(parents=True,exist_ok=False)
    schema=json.loads((ROOT/'data/metadata/feature_schema.json').read_text())
    seq=np.load(ROOT/'data/model_input/financial_sequences.npz',allow_pickle=False)
    labels=read(ROOT/'data/metadata/labels.csv')
    graph=np.load(ROOT/'data/model_input/structural_baseline/quarter_graphs.npz',allow_pickle=False)
    labels['qindex']=labels.feature_quarter.map({q:i for i,q in enumerate(graph['quarters'])})
    frames={s:labels[labels.split.eq(s)].sort_values(['qindex','firm_id']).reset_index(drop=True) for s in ['Train','Valid','Test']}
    ends={s:sorted(f.qindex.unique()) for s,f in frames.items()}
    sx=torch.from_numpy(np.load(ROOT/'data/model_input/structural_baseline/node_features.npy'))
    pa=torch.from_numpy(graph['patent']);ia=torch.from_numpy(graph['ipc'])
    rows=[]
    for name in ['1D_CNN','LSTM','GRU','structural_adaptation']:
        structural=name=='structural_adaptation'
        y={s:torch.tensor((frames[s].distress_label.to_numpy() if structural else seq['distress_label'][seq['split']==s]),dtype=torch.float32) for s in frames}
        x={s:torch.tensor(seq['x'][seq['split']==s].astype(np.float32)) for s in frames}
        weight=torch.tensor(2100/236) if structural else torch.tensor([2100/236],dtype=torch.float32)
        for seed in schema['formal_seeds']:
            seed_all(seed)
            model=Net(sx.shape[-1]) if structural else CNN() if name=='1D_CNN' else RNN(nn.LSTM if name=='LSTM' else nn.GRU)
            optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0001)
            def forward(split):return model.logits(sx,pa,ia,ends[split]) if structural else model(x[split])
            best=None;stale=0
            for epoch in range(1,301):
                model.train();optimizer.zero_grad()
                loss=F.binary_cross_entropy_with_logits(forward('Train'),y['Train'],pos_weight=weight)
                loss.backward();optimizer.step();model.eval()
                with torch.no_grad():
                    valid_logits=forward('Valid')
                    valid_loss=float(F.binary_cross_entropy_with_logits(valid_logits,y['Valid'],pos_weight=weight))
                    probability=torch.sigmoid(valid_logits).numpy()
                key=(-valid_loss,) if structural else selector(y['Valid'].numpy(),probability,historical=True)[1]
                if best is None or key>best[0]:
                    best=(key,copy.deepcopy(model.state_dict()));stale=0;best_epoch=epoch
                else:stale+=1
                if stale>=30:break
            model.load_state_dict(best[1]);model.eval()
            with torch.no_grad():
                logits=forward('Valid')
                vp=torch.sigmoid(logits).numpy() if structural else 1/(1+np.exp(-logits.double().numpy()))
            if structural:
                threshold=max([i/100 for i in range(5,96)],key=lambda t:(metrics(y['Valid'].numpy(),vp,t)['f1'],metrics(y['Valid'].numpy(),vp,t)['accuracy'],-t))
            else:threshold=selector(y['Valid'].numpy(),vp)[0]
            torch.save({'state_dict':best[1],'seed':seed,'best_epoch':best_epoch,'threshold':threshold},out/f'{name}_seed_{seed}.pt')
            with torch.no_grad():
                logits=forward('Test')
                probability=torch.sigmoid(logits).numpy() if structural else 1/(1+np.exp(-logits.double().numpy()))
            value=metrics(y['Test'].numpy(),probability,threshold)
            rows.append({'model':name,'seed':seed,**{k:value[k] for k in ['precision','recall','f1','accuracy']}})
            pd.DataFrame(rows).to_csv(out/'per_seed_metrics.csv',index=False)

if __name__=='__main__':main()
