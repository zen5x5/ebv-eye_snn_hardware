"""Evaluate selected 24-user state-depth quantization groups on CPU."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import multiprocessing
from pathlib import Path
import platform
import shutil
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
VARIANTS = ('full_state', 'l1_stateless', 'l1_l2_stateless', 'l1_l2_l3_stateless',
            'l1_l2_l3_l4_stateless')
TASKS = ('pursuit', 'saccade')
METRICS = ('mean_angular_error_deg', 'mean_pixel_error')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def prepare(args):
    out = args.output_dir.resolve()
    out.mkdir(parents=True)
    rows = []
    for task in args.tasks:
        for variant_index, variant in enumerate(args.variants):
            depth = VARIANTS.index(variant)
            for user in range(4, 28):
                ordinary = 'ordinary_' if task == 'saccade' else ''
                experiment = f'all_physical_source_recombination_{ordinary}t2_alpha0p75_{variant}_continue_epoch100_to200'
                source = args.models_root/task/'fixed_event_window/snn'/f'user{user}'/'eye0'/experiment/'seed2021'
                cache = args.cache_root/task/'fixed_event_window_400_stride200_96x128'/f'user{user}'/'eye0'
                config = json.loads((source/'configuration.json').read_text())
                summary = json.loads((source/'summary.json').read_text())
                expected = {f'L{i+1}': list(range((8, 16, 24, 32)[i])) for i in range(depth)}
                assert config['training']['seed'] == 2021 and config['training']['epochs'] == 200
                assert config['input']['temporal_steps'] == 2
                assert config['input']['temporal_organization'] == 'repeat_identical_complete_400_event_count_frame'
                assert config['model']['stateless_channels'] == expected
                assert summary['trainable_parameters'] == 96792
                assert config['model']['alpha'] == 0.75
                float_metrics = summary['test'] if task == 'pursuit' else summary['metrics']['test']
                name = f'{task}/{variant}/user{user}'
                folder = out/'subjects'/name
                folder.mkdir(parents=True)
                frozen = out/'inputs/models'/name
                frozen.mkdir(parents=True)
                hashes = {}
                for filename in ('configuration.json', 'summary.json', 'best_checkpoint.pt'):
                    original = source/filename
                    hashes[str(original)] = sha(original)
                    shutil.copy2(original, frozen/filename)
                    assert sha(frozen/filename) == hashes[str(original)]
                local_cache = out/'inputs/test'/task/f'user{user}'
                if variant_index == 0:
                    local_cache.mkdir(parents=True)
                    for filename in ('manifest.json', 'test.npz'):
                        shutil.copy2(cache/filename, local_cache/filename)
                        assert sha(cache/filename) == sha(local_cache/filename)
                for filename in ('manifest.json', 'test.npz'):
                    hashes[str(cache/filename)] = sha(local_cache/filename)
                if task == 'saccade':
                    assert config['cache_sha256']['test.npz'] == hashes[str(cache/'test.npz')]
                manifest = json.loads((local_cache/'manifest.json').read_text())
                assert manifest['splits']['test']['sha256'] == hashes[str(cache/'test.npz')]
                row = dict(case=name, task=task, variant=variant, depth=depth, user=user,
                           representations=int(float_metrics['samples']),
                           official_float={key: float(float_metrics[key]) for key in METRICS},
                           input_sha256=hashes, model=str(frozen.relative_to(out)),
                           test=str((local_cache/'test.npz').relative_to(out)),
                           output=str(folder.relative_to(out)), execution='pending')
                if depth == 2:
                    retained = ROOT/'results/quantization/multiuser'/task/f'user{user}'/'summary.json'
                    old = json.loads(retained.read_text())
                    for original, expected_hash in hashes.items():
                        assert old['input_sha256'][original] == expected_hash
                    row.update(execution='retained', retained_source=str(retained.relative_to(ROOT)),
                               retained_sha256=sha(retained), fixed_point=old['fixed_point']['continuous'],
                               runtime_counters=old['runtime_counters'])
                    shutil.copy2(retained, folder/'retained_source.json')
                    write_json(folder/'summary.json', row)
                elif task == 'pursuit' and user == 10 and depth < 4:
                    retained = ROOT/'results/quantization/user10'/f'{variant}.json'
                    old = json.loads(retained.read_text())
                    assert old['checkpoint_sha256'] == hashes[str(source/'best_checkpoint.pt')]
                    assert old['configuration_sha256'] == hashes[str(source/'configuration.json')]
                    assert old['test_cache_sha256'] == hashes[str(cache/'test.npz')]
                    assert old['test_samples'] == row['representations']
                    row.update(execution='retained', retained_source=str(retained.relative_to(ROOT)),
                               retained_sha256=sha(retained), fixed_point=old['fixed_test'],
                               runtime_counters=old['runtime_counters'])
                    shutil.copy2(retained, folder/'retained_source.json')
                    write_json(folder/'summary.json', row)
                rows.append(row)
    configuration = dict(seed=2021, tasks=args.tasks, users=list(range(4, 28)), variants=args.variants,
                         device='cpu', split='test', output_encoding='continuous_xy',
                         float_source='original L40S/Exodus best-validation checkpoint reports',
                         aggregation='mean within subject, then equal mean across 24 subjects',
                         models_root=str(args.models_root), cache_root=str(args.cache_root),
                         script_sha256=sha(__file__), host=platform.node())
    write_json(out/'configuration.json', configuration)
    write_json(out/'input_manifest.json', rows)
    print(json.dumps(dict(status='prepared', models=len(rows), retained=sum(r['execution']=='retained' for r in rows),
                          pending=sum(r['execution']=='pending' for r in rows))), flush=True)


def initialize(threads):
    import torch
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)


def evaluate_one(out_name, row):
    import numpy as np
    import torch
    from hardware.quantization.fixed_point import load_model, RuntimeCounters
    from hardware.quantization.formal import CONTRACT, ExactFixedPointReference, errors, means, normalized_to_pixels
    from hardware.quantization.integer_reference import reference as integer_reference
    started = time.monotonic()
    out = Path(out_name)
    folder = out/row['output']
    assert not (folder/'summary.json').exists(), row['case']
    model_dir = out/row['model']
    config = json.loads((model_dir/'configuration.json').read_text())
    model, checkpoint = load_model(model_dir/'best_checkpoint.pt', config)
    stateless = tuple(f'L{i+1}' for i in range(row['depth']))
    reference = ExactFixedPointReference(model, **CONTRACT, stateless_layer_names=stateless)
    with np.load(out/row['test'], allow_pickle=False) as data:
        frames = data['event_frames'].copy()
        labels = data['labels'].astype(np.float64)
    assert len(frames) == row['representations'] and frames.shape[1:] == (2, 96, 128)
    assert np.all(frames.sum(axis=(1, 2, 3), dtype=np.uint64) == 400)
    parameters = reference.parameter_arrays()
    probe_trace = {}
    with torch.inference_mode():
        reference.predict(torch.from_numpy(frames[:1]).float(), RuntimeCounters.empty(), probe_trace)
    independent = integer_reference(parameters, frames[:1], stateless)
    for key, expected in independent.items():
        np.testing.assert_array_equal(expected, probe_trace[key], err_msg=row['case']+' '+key)
    counters = RuntimeCounters.empty()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(frames), 16):
            predictions.append(reference.predict(torch.from_numpy(frames[start:start+16]).float(), counters).numpy())
    prediction = np.concatenate(predictions)
    pixels = normalized_to_pixels(prediction)
    angular, distance = errors(pixels, labels)
    np.savez_compressed(folder/'predictions_test.npz', prediction_normalized_xy=prediction,
                        prediction_row_col_px=pixels, target_row_col_px=labels,
                        angular_error_deg=angular, pixel_error=distance,
                        representation_index=np.arange(len(frames)))
    np.savez_compressed(folder/'parameters.npz', **parameters)
    result = dict(row, execution='new_cpu', checkpoint_epoch=int(checkpoint['epoch']),
                  fixed_point=means((angular, distance)), runtime_counters=asdict(counters),
                  arithmetic_contract=CONTRACT, state_bits=16,
                  independent_probe_arrays=len(independent),
                  cpu_threads=torch.get_num_threads(), host=platform.node(),
                  torch=str(torch.__version__), numpy=np.__version__,
                  runtime_seconds=time.monotonic()-started)
    write_json(folder/'summary.json', result)
    print(json.dumps(dict(case=row['case'], samples=len(frames),
                          fixed_deg=result['fixed_point']['mean_angular_error_deg'],
                          seconds=result['runtime_seconds'])), flush=True)
    return row['case']


def evaluate(args):
    out = args.output_dir.resolve()
    rows = json.loads((out/'input_manifest.json').read_text())
    pending = [r for r in rows if r['execution']=='pending']
    if args.only:
        pending = [r for r in pending if r['case'] in args.only]
        assert {r['case'] for r in pending} == set(args.only)
    if args.exclude:
        assert set(args.exclude) <= {r['case'] for r in pending}
        pending = [r for r in pending if r['case'] not in args.exclude]
    print('EVALUATING',len(pending),'models,',args.workers,'workers x',args.threads,'CPU threads',flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=initialize, initargs=(args.threads,)) as pool:
        jobs = [pool.submit(evaluate_one, str(out), row) for row in pending]
        for index, future in enumerate(as_completed(jobs), 1):
            print('PROGRESS',index,len(jobs),future.result(),flush=True)


def summarize(args):
    out = args.output_dir.resolve()
    configuration = json.loads((out/'configuration.json').read_text())
    manifest = json.loads((out/'input_manifest.json').read_text())
    rows = [json.loads((out/r['output']/'summary.json').read_text()) for r in manifest]
    aggregates = []
    for task in configuration['tasks']:
        for variant in configuration['variants']:
            group = [r for r in rows if r['task']==task and r['variant']==variant]
            assert len(group)==24 and {r['user'] for r in group}==set(range(4,28))
            aggregates.append(dict(task=task, variant=variant, subjects=24,
                                   representations=sum(r['representations'] for r in group),
                                   official_float={k: statistics.mean(r['official_float'][k] for r in group) for k in METRICS},
                                   fixed_point={k: statistics.mean(r['fixed_point'][k] for r in group) for k in METRICS}))
    write_json(out/'results.json',dict(status='complete',models=len(rows),
                                     new_models=sum(r['execution']=='new_cpu' for r in rows),
                                     retained_models=sum(r['execution']=='retained' for r in rows),
                                     aggregates=aggregates,subjects=rows))
    for row in aggregates:
        print(json.dumps(row),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('prepare','evaluate','summarize'))
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--models-root',type=Path)
    parser.add_argument('--cache-root',type=Path)
    parser.add_argument('--tasks',nargs='+',choices=TASKS,default=TASKS)
    parser.add_argument('--variants',nargs='+',choices=VARIANTS,default=VARIANTS[:4])
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--only',nargs='+')
    parser.add_argument('--exclude',nargs='+')
    arguments = parser.parse_args()
    {'prepare':prepare,'evaluate':evaluate,'summarize':summarize}[arguments.operation](arguments)
