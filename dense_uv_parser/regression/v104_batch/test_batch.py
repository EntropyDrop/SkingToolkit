"""File-safety and interruption recovery checks; no GPU or real dataset writes."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from PIL import Image


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


HERE = Path(__file__).resolve().parent
launcher = module('launcher', HERE.parents[1] / 'run_all_edited_v104.py')
runner = module('runner', HERE / 'runner.py')


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def png(self, path, color=(30, 60, 90, 255)):
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGBA', (64, 64), color).save(path)
        return path

    def test_recursive_discovery_ignores_other_versions_and_keeps_directory_identity(self):
        for relative in ['a/X_edited.png', 'b/X_edited.png', 'a/X_source.png',
                         'a/X_result_v103.png', 'a/X_result_v104.png', 'a/X_edited_preview.png']:
            self.png(self.root / relative)
        rows = launcher.discover(self.root)
        self.assertEqual(len(rows), 2)
        self.assertEqual({Path(r['target']).relative_to(self.root).as_posix() for r in rows},
                         {'a/X_result_v104.png', 'b/X_result_v104.png'})
        self.assertEqual(sum(r['previous_result_sha256'] is not None for r in rows), 1)

    def test_concurrent_target_change_is_not_overwritten(self):
        artifact = self.png(self.root/'artifact.png')
        target = self.png(self.root/'target.png', (200, 0, 0, 255))
        old = target.read_bytes()
        with self.assertRaisesRegex(ValueError, 'Another writer'):
            runner.atomic_export(artifact, target, 'different-original-hash')
        self.assertEqual(target.read_bytes(), old)
        self.assertEqual(list(self.root.glob('.*.tmp')), [])

    def test_blank_artifact_does_not_replace_existing_output(self):
        artifact = self.png(self.root/'blank.png', (0, 0, 0, 0))
        target = self.png(self.root/'target.png')
        old = target.read_bytes()
        with self.assertRaisesRegex(ValueError, 'transparent'):
            runner.atomic_export(artifact, target, runner.sha(target))
        self.assertEqual(target.read_bytes(), old)

    def test_recover_completed_artifact_and_truncated_journal_without_inference(self):
        project = self.root/'project'
        (project/'dense_uv_parser/runs').mkdir(parents=True)
        job = project/'dense_uv_parser/runs/job'
        artifact = job/'artifacts/sample'
        artifact.mkdir(parents=True)
        source = self.png(self.root/'dataset/X_edited.png')
        target = self.png(self.root/'dataset/X_result_v104.png', (100, 0, 0, 255))
        backup = self.root/'backup.png'
        backup.write_bytes(target.read_bytes())
        prediction = self.png(artifact/'parser_pred_uv_simple_inpainting.png', (10, 20, 30, 255))
        item = {'source':str(source), 'source_sha256':runner.sha(source), 'target':str(target),
                'previous_result_sha256':runner.sha(target), 'backup':str(backup)}
        config = {'final_head_material_refine_steps':64}
        settings = {'pinned_sha256':{}, 'workers':2, 'checkpoint_sha256':'model',
                    'pipeline_sha256':'pipeline', 'foreground_sha256':'foreground',
                    'parser_source_sha256':{}, 'data_root':str(source.parent)}
        manifest = {'complete':True, 'input':str(source), 'input_sha256':runner.sha(source),
                    'checkpoint_sha256':'model', 'pipeline':config, 'source_sha256':{},
                    'foreground_provider':{'adaptation_sha256':'foreground'}}
        for path, value in [(job/'job.json',settings), (job/'pipeline.json',config),
                            (job/'inputs.json',[item]), (job/'protected_files.json',
                            [{'path':str(backup),'sha256':runner.sha(backup)}]),
                            (artifact/'manifest.json',manifest),
                            (artifact/'final_head_material_refit.json',{'alpha_exact':True,'body_exact':True})]:
            path.write_text(json.dumps(value))
        with mock.patch.object(runner, 'PROJECT', project), mock.patch.object(runner, 'JOB', job), \
             mock.patch.object(runner.signal, 'signal'), \
             mock.patch.object(runner.subprocess, 'Popen', side_effect=AssertionError('Should recover without GPU work')):
            self.assertEqual(runner.main(), 0)
            self.assertEqual(target.read_bytes(), prediction.read_bytes())
            with (job/'results.jsonl').open('ab') as f:
                f.write(b'{"source": "interrupted')
            self.assertEqual(runner.main(), 0)
        progress = json.loads((job/'progress.json').read_text())
        self.assertEqual(progress['initially_complete'], 1)
        self.assertEqual(progress['completed'], 1)
        self.assertEqual(len((job/'results.jsonl').read_text().splitlines()), 1)
        self.assertEqual(runner.sha(backup), item['previous_result_sha256'])
        self.assertEqual(runner.sha(source), item['source_sha256'])


if __name__ == '__main__':
    unittest.main()
