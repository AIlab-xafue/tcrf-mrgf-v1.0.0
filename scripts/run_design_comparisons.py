from pathlib import Path
import subprocess,sys
if __name__=='__main__':
    for design in ['direct_technology','text_patent','patent_only','jaccard_ipc']:
        subprocess.run([sys.executable,str(Path(__file__).with_name('run_main_experiment.py')),'--design',design],check=True)
