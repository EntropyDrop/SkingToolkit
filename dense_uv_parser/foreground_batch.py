"""Run the trained foreground model and UV parser with their existing environments."""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--foreground-checkpoint', type=Path, required=True)
    p.add_argument('--foreground-model-dir', type=Path, required=True)
    p.add_argument('--foreground-python', default='/home/ds/miniconda3/envs/comfy/bin/python')
    p.add_argument('--parser-python', default='/home/ds/miniconda3/envs/sking-v61-worker/bin/python')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--probability-dir', type=Path, required=True)
    p.add_argument('--pipeline', type=Path)
    p.add_argument('--inputs', type=Path, nargs='+')
    p.add_argument('--save-tensors', action='store_true')
    opt = p.parse_args()
    inputs = opt.inputs or [Path(x) for x in json.loads((root/'dense_uv_parser/v101_cases.json').read_text())]
    for path in inputs + [opt.checkpoint, opt.foreground_checkpoint]:
        if not path.is_file():
            raise FileNotFoundError(path)
    entry = root/'dense_uv_parser/run_local.py'
    with tempfile.TemporaryDirectory(prefix='foreground-batch-') as tmp:
        manifest = Path(tmp)/'results.json'
        subprocess.run([opt.foreground_python, str(entry), 'foreground_provider',
                        '--checkpoint', str(opt.foreground_checkpoint.resolve()),
                        '--model-dir', str(opt.foreground_model_dir.resolve()),
                        '--output-dir', str(opt.probability_dir.resolve()),
                        '--result-manifest', str(manifest),
                        '--inputs', *[str(x.resolve()) for x in inputs]], cwd=root, check=True)
        probability = json.loads(manifest.read_text())['probabilities']
        if len(probability) != len(inputs):
            raise ValueError('Foreground output count does not match inputs')
        command = [opt.parser_python, str(entry), 'batch_accessories',
                   '--checkpoint', str(opt.checkpoint.resolve()),
                   '--output-dir', str(opt.output_dir.resolve()),
                   '--foreground-probabilities', *probability,
                   '--inputs', *[str(x.resolve()) for x in inputs]]
        if opt.pipeline:
            command += ['--pipeline', str(opt.pipeline.resolve())]
        if opt.save_tensors:
            command += ['--save-tensors']
        subprocess.run(command, cwd=root, check=True)


if __name__ == '__main__':
    main()
