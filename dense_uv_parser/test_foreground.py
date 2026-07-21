import sys
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from SkingToolkit.dense_uv_parser.foreground import (
    build_parser_input,
    save_flood_outputs,
)


class DenseParserForegroundTest(unittest.TestCase):
    def test_run_infer_defines_versioned_checkpoint_lookup_before_use(self):
        script = (Path(__file__).parent / "run_infer.sh").read_text(
            encoding="utf-8"
        )
        definition = script.index("find_latest_checkpoint()")
        parser_lookup = script.index(
            'PARSER_CHECKPOINT="$(find_latest_checkpoint '
        )
        self.assertLess(definition, parser_lookup)

    def test_dense_infer_does_not_import_removed_foreground_package(self):
        import SkingToolkit.dense_uv_parser.infer  # noqa: F401

        self.assertNotIn("SkingToolkit.fixed_view_foreground", sys.modules)
        self.assertNotIn("SkingToolkit.fixed_view_foreground.inference", sys.modules)

    def test_parser_input_preserves_foreground_and_replaces_background(self):
        rendered = torch.zeros(1, 4, 16, 16)
        rendered[:, :3] = 0.5
        rendered[:, 3] = 1.0
        rendered[:, :3, 4:12, 4:12] = torch.tensor(
            [0.8, 0.2, 0.4]
        ).view(1, 3, 1, 1)
        foreground = torch.zeros(1, 16, 16, dtype=torch.bool)
        foreground[:, 4:12, 4:12] = True

        parser_input = build_parser_input(rendered, foreground)

        self.assertTrue(
            torch.equal(
                parser_input[0, :3, 6, 6], rendered[0, :3, 6, 6]
            )
        )
        self.assertFalse(
            torch.equal(
                parser_input[0, :3, 0, 0], rendered[0, :3, 0, 0]
            )
        )
        self.assertTrue(torch.all(parser_input[:, 3] == 1.0))

    def test_flood_cutout_has_transparent_rejected_background(self):
        rendered = torch.rand(1, 4, 8, 8)
        rendered[:, 3] = 1.0
        foreground = torch.zeros(1, 8, 8, dtype=torch.bool)
        foreground[:, 2:6, 2:6] = True
        with tempfile.TemporaryDirectory() as temporary:
            cutout_path = Path(temporary) / "cutout.png"
            returned = save_flood_outputs(
                rendered,
                foreground,
                view_count=1,
                cutout_output=cutout_path,
            )
            cutout = Image.open(cutout_path).convert("RGBA")

        self.assertTrue(torch.equal(returned, foreground))
        self.assertEqual(cutout.getpixel((0, 0))[3], 0)
        self.assertEqual(cutout.getpixel((3, 3))[3], 255)


if __name__ == "__main__":
    unittest.main()
