"""Export paper tables from retained results; no inference or FPGA tools run."""
import argparse
import csv
import json
from pathlib import Path
from _shared import ROOT


def rows():
    summary = json.loads((ROOT/'results/summary.json').read_text())
    output = []
    for variant in summary['variants']:
        name = variant['variant']
        quant = json.loads((ROOT/'results/quantization/user10'/f'{name}.json').read_text())
        row = dict(variant=name, user=10, seed=2021,
                   float_test_deg=quant['float_test']['mean_angular_error_deg'],
                   fixed_test_deg=quant['fixed_test']['mean_angular_error_deg'],
                   cycles=variant['performance']['core_cycles'], frequency_mhz=160,
                   core_latency_ms=variant['performance']['core_latency_ms'],
                   core_samples_per_second=variant['performance']['core_service_rate_samples_per_second'])
        for resource in ('lut', 'ff', 'dsp', 'bram36_equivalent'):
            row[resource] = variant['resources'][resource]['used']
            row[resource+'_percent'] = variant['resources'][resource]['percent']
        for kind in ('total_w', 'dynamic_w', 'static_w', 'total_energy_uj', 'dynamic_energy_uj', 'static_energy_uj'):
            row[kind] = variant['power']['metrics'][kind]['mean']
            row[kind+'_std'] = variant['power']['metrics'][kind]['population_std']
        row.update(variant['relative_to_full_state'])
        output.append(row)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    records = rows()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f'Exported {len(records)} variants to {args.output}')


if __name__ == '__main__':
    main()
