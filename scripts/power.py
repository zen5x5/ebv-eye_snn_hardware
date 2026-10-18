"""Re-estimate power from retained activity, without re-running simulation."""
import argparse
import gzip
from pathlib import Path
import shutil
from _shared import ROOT, VARIANTS, run, tool


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True)
    activities = out/'activity'
    activities.mkdir()
    for sample in range(16):
        name = f'sample_{sample:02d}'
        folder = activities/name
        folder.mkdir()
        source = ROOT/'results/power'/args.variant/name/'inference.saif.gz'
        with gzip.open(source, 'rb') as incoming, (folder/'inference.saif').open('wb') as outgoing:
            shutil.copyfileobj(incoming, outgoing)
    run([tool('vivado'), '-mode', 'batch', '-nojournal', '-log', out/'vivado.log',
         '-source', ROOT/'hardware/vivado/power.tcl', '-tclargs',
         ROOT/'results/implementation'/args.variant/'post_route.dcp', activities, out/'reports'], out/'console.log', ROOT)


if __name__ == '__main__':
    main()
