"""Check task-independent cycles, arithmetic layout and runtime-loaded weights."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts._shared import VARIANTS, CYCLES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    replay = json.loads(args.replay.read_text())
    assert replay['status']=='passed'
    baseline = json.loads((ROOT/'results/rtl/report.json').read_text())['variants']
    implementation = json.loads((ROOT/'results/summary.json').read_text())['variants']
    checks=[]
    for depth, variant in enumerate(VARIANTS[:4]):
        for user in range(4,28):
            configs=[]
            for task in ('pursuit','saccade'):
                path=ROOT/'results/quantization/state_depth/inputs/models'/task/variant/f'user{user}/configuration.json'
                config=json.loads(path.read_text())
                assert config['input']['temporal_steps']==2
                assert config['input']['temporal_organization']=='repeat_identical_complete_400_event_count_frame'
                assert config['model']['alpha']==0.75
                assert config['model']['minimum_membrane_potential']==-1
                configs.append(config)
            assert configs[0]['model']['architecture']==configs[1]['model']['architecture']
            assert configs[0]['model']['conv_channels']==configs[1]['model']['channels']
            assert configs[0]['model']['stateless_channels']==configs[1]['model']['stateless_channels']
            assert configs[0]['model']['retained_cross_bin_membrane_states']==configs[1]['model']['retained_cross_bin_membrane_states']
            assert configs[1]['model']['output_encoding']=='continuous_xy'
        case=args.cases/f'saccade_{variant}'
        metadata=json.loads((case/'summary.json').read_text())
        with np.load(ROOT/'models'/variant/'parameters.npz',allow_pickle=False) as pursuit, np.load(case/'parameters.npz',allow_pickle=False) as saccade:
            assert set(pursuit.files)==set(saccade.files)
            assert all(pursuit[k].shape==saccade[k].shape and pursuit[k].dtype==saccade[k].dtype for k in pursuit.files)
            assert not np.array_equal(pursuit['readout.weight_integer'],saccade['readout.weight_integer'])
        original_layout=json.loads((ROOT/'models'/variant/'hardware_image/layout.json').read_text())
        task_layout=json.loads((case/'hardware_image/layout.json').read_text())
        for key in ('convolution','linear','channel_parameters','readout'):
            assert original_layout[key]==task_layout[key]
        selected=[r for r in replay['records'] if r['depth']==depth]
        assert len(selected)==16 and all(r['task']=='saccade' and r['mismatches']==0 for r in selected)
        assert all(r['core_cycles']==CYCLES[depth]==baseline[variant]['core_cycles'] for r in selected)
        original_layers=baseline[variant]['records'][0]['layers']
        for row in selected:
            actual=[[r[k] for k in ('layer','cycles','state_write_groups','state_read_groups','mac_issue_groups')] for r in row['layers']]
            assert actual==original_layers
        quant=json.loads((ROOT/'results/quantization/user10'/f'{variant}.json').read_text())
        assert quant['arithmetic_contract']==metadata['arithmetic']
        shared_dcp=ROOT/'results/implementation'/variant/'post_route.dcp'
        checks.append(dict(variant=variant,models_audited=48,saccade_rtl_samples=16,
                           core_cycles=CYCLES[depth],latency_ms=CYCLES[depth]/160000,
                           bram36_equivalent=implementation[depth]['resources']['bram36_equivalent']['used'],
                           same_cycle_count=True,same_layer_cycles_and_memory_access_counts=True,
                           same_parameter_shapes_and_hardware_image_layout=True,
                           separately_trained_weights_confirmed=True,
                           common_implementation_checkpoint=str(shared_dcp.relative_to(ROOT)),
                           common_implementation_sha256=hashlib.sha256(shared_dcp.read_bytes()).hexdigest()))
    sources=list((ROOT/'hardware/rtl').glob('*.sv'))
    assert all('$readmemh' not in path.read_text() and '$readmemb' not in path.read_text() for path in sources)
    report=dict(status='passed',scope='Table II continuous_xy Saccade versus Pursuit, same state-depth configuration',
                trained_model_configurations_checked=192,saccade_rtl_samples_checked=64,
                runtime_loaded_weights=True,task_specific_synthesized_logic=False,
                same_latency_and_bram=True,comparisons=checks,
                fixed_schedule='spike values change arithmetic data, not iteration bounds or memory capacity',
                l1_l4_rule='the depth-4 implementation is also shared when network dimensions, arithmetic and stateless depth match')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('status','trained_model_configurations_checked',
                                          'saccade_rtl_samples_checked','same_latency_and_bram')}))


if __name__=='__main__':
    main()
