"""Validate packaged assets without model training or inference."""
from pathlib import Path
import ast, json, re, sys
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]

def verify():
    checks = {}
    paths = [p for p in ROOT.rglob('*') if '__pycache__' not in p.parts and 'outputs' not in p.parts]
    checks['english_filenames'] = all(str(p.relative_to(ROOT)).isascii() for p in paths)
    issues = []
    # Construct scan patterns from fragments so the validator does not flag its own vocabulary.
    forbidden = ['hg'+'smo'+'te', 'smo'+'te', 'application'+'date', 'grant'+'pubdate', 'filing'+'_date', 'publication'+'_date', 'grant'+'_date', 'application'+'_date', 'error'+'_transition', 'persistent'+'_fn', 'case'+'_evidence', 'selected'+'_case', 'margin'+'_analysis', 'distress_type'+'_analysis', 'fn'+'_to_tp', 'fp'+'_to_tn']
    branch = re.compile(r'(^|[_/\\.-])(old|back'+'up|pi'+'lot|t'+'mp|syn'+'thetic)([_/\\.-]|$)',re.I)
    for p in paths:
        if not p.is_file():continue
        rel = p.relative_to(ROOT).as_posix()
        if branch.search(rel):issues.append([rel,'excluded artifact name'])
        if p.suffix not in {'.py','.md','.yaml','.json','.csv','.txt'}:continue
        text = p.read_text(encoding='utf-8-sig')
        if re.search('[\u3400-\u9fff]', text):issues.append([rel,'non-English text'])
        if any(term in text.lower() or term in rel.lower() for term in forbidden):issues.append([rel,'excluded terminology'])
        if re.search(r'[A-Za-z]:[\\/]',text):issues.append([rel,'absolute path'])
        if re.search(r'(?:sk-[A-Za-z0-9]{20,}|(?:api_key|password|token)\s*[:=]\s*[\"\x27][A-Za-z0-9_-]{16,})',text):issues.append([rel,'credential pattern'])
        if p.suffix=='.py':ast.parse(text)
    checks['content_scan'] = not issues
    def read(name):return pd.read_csv(ROOT/name,dtype={'firm_id':str,'firm_a':str,'firm_b':str},float_precision='round_trip')
    labels = read('data/metadata/labels.csv')
    node = read('data/model_input/node_features.csv')
    graph = read('data/graph/joint_edges.csv')
    schema = json.loads((ROOT/'data/metadata/feature_schema.json').read_text())
    checks['sample_identity'] = len(labels)==3504 and labels.firm_id.nunique()==146 and not labels.duplicated(['firm_id','feature_quarter']).any()
    checks['quarters'] = sorted(labels.feature_quarter.unique())==list(pd.period_range('2018Q4','2024Q3',freq='Q').astype(str))
    checks['splits'] = labels.groupby('split').size().to_dict()=={'Train':2336,'Valid':584,'Test':584} and labels.groupby('split').distress_label.sum().to_dict()=={'Train':236,'Valid':64,'Test':69}
    features = schema['finance_features']+schema['tcrf_features']
    checks['features'] = len(schema['finance_features'])==25 and len(features)==27 and np.isfinite(node[features]).all().all()
    keys = ['firm_id','feature_quarter','target_quarter','split']
    checks['node_order'] = labels[keys].equals(node[keys])
    node_keys = set(zip(labels.feature_quarter,labels.firm_id))
    checks['graph'] = len(graph)==1323 and graph.relation_type.value_counts().to_dict()=={'IPC':882,'Patent':441} and all((r.feature_quarter,r.firm_a) in node_keys and (r.feature_quarter,r.firm_b) in node_keys for r in graph.itertuples())
    expected_seeds = [1,7,11,17,21,33,42,77,1001,2024]
    training = yaml.safe_load((ROOT/'configs/formal_training.yaml').read_text())
    checks['seeds'] = training['formal_seeds']==schema['formal_seeds']==expected_seeds
    model = yaml.safe_load((ROOT/'configs/formal_model.yaml').read_text())
    checks['architecture'] = all(model[k]==v for k,v in {'input_dim':27,'node_hidden_dims':[64,64],'gat_layers':1,'gat_heads':4,'graph_output_dim':32,'classifier_dims':[32,64,1]}.items())
    checks['weighted_bce'] = training['loss']=='weighted_bce' and training['optimizer']=='adam' and training['learning_rate']==.001
    results = read('results/ablation/module_ablation_per_seed.csv')
    checks['result_seeds'] = len(results)==40 and all(sorted(g.seed)==expected_seeds for _,g in results.groupby('configuration'))
    checks['result_signature'] = bool(np.allclose(results.groupby('configuration',sort=False).f1.mean(),[.633356,.656102,.662981,.668667],atol=1e-6,rtol=0))
    main = results[results.configuration.eq('M3_FULL')][['precision','recall','f1','accuracy']].mean()
    checks['full_signature'] = bool(np.allclose(main,[.589144,.775362,.668667,.909247],atol=1e-6,rtol=0))
    print(json.dumps({'gate':'PASS' if all(checks.values()) else 'FAIL','checks':{k:bool(v) for k,v in checks.items()},'issues':issues},indent=2))
    return all(checks.values())

if __name__=='__main__':
    sys.exit(0 if verify() else 1)
