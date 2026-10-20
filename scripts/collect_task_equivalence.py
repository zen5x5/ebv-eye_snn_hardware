"""Retain exact task-comparison vectors, logs and their independent checks."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from _shared import ROOT, VARIANTS


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-directory',type=Path,required=True)
    args=parser.parse_args()
    run=args.run_directory.resolve()
    source=run/'rtl_saccade'
    replay=json.loads((source/'summary.json').read_text())
    assert replay['status']=='passed' and len(replay['records'])==64
    target=ROOT/'results/task_equivalence'
    target.mkdir(parents=True,exist_ok=False)
    shutil.copy2(source/'summary.json',target/'replay.json')
    shutil.copytree(source/'source_snapshot',target/'source_snapshot')
    for variant in VARIANTS[:4]:
        name=f'saccade_{variant}'
        case=run/'cases'/name
        output=target/'cases'/name
        output.mkdir(parents=True)
        for filename in ('summary.json','parameters.npz','inputs.npz','trace.npz'):
            shutil.copy2(case/filename,output/filename)
        shutil.copytree(case/'hardware_image',output/'hardware_image')
        with tarfile.open(output/'sample_memories.tar.gz','w:gz') as archive:
            for sample in range(16):
                archive.add(case/f'sample_{sample:02d}',arcname=f'sample_{sample:02d}')
        for sample in range(16):
            log=target/'logs'/name/f'sample_{sample:02d}.log'
            log.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source/name/f'sample_{sample:02d}/simulation.log',log)
    subprocess.run([sys.executable,ROOT/'scripts/verify_task_equivalence.py',
                    '--cases',target/'cases','--replay',target/'replay.json',
                    '--output',target/'verification.json'],check=True)


if __name__=='__main__':
    main()
