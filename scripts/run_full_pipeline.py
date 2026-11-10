"""Run the released formal experiments explicitly."""
from pathlib import Path
import subprocess,sys
if __name__=='__main__':
    folder=Path(__file__).resolve().parent
    root=folder.parent
    for path in [folder/'verify_release.py',root/'src/features/build_tcrf.py',root/'src/graph/build_multirelational_graph.py',folder/'run_financial_baselines.py',folder/'run_neural_baselines.py',folder/'run_module_ablation.py',folder/'run_main_experiment.py',folder/'run_design_comparisons.py']:
        subprocess.run([sys.executable,str(path)],check=True)
