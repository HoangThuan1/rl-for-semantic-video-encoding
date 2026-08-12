import json
import os
import sys
import tempfile
import unittest
from unittest import mock


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "rl_env_offline"))

import encode_grid  # noqa: E402
import export_encoded_video  # noqa: E402
from env import EncoderAction, RESOLUTIONS, ROI_QOFFSET_LEVELS, VCUSimEnv  # noqa: E402


class RoiGeometryTests(unittest.TestCase):
    def test_each_detection_becomes_an_aligned_roi(self):
        metadata = {
            0: {
                "boxes": [
                    [1, 3, 21, 23],
                    [40, 10, 80, 50],
                    [1, 3, 21, 23],
                ]
            }
        }

        boxes = encode_grid.roi_boxes_for_frames(
            metadata, 0, 1, 100, 100, 200, 100
        )

        self.assertEqual(boxes, [(2, 2, 40, 22), (80, 10, 80, 40)])

    def test_filter_chain_has_one_addroi_per_box(self):
        chain = encode_grid.addroi_filter_chain(
            [(2, 4, 20, 30), (40, 50, 60, 70)], -0.6
        )

        self.assertEqual(chain.count(",addroi="), 2)
        self.assertIn("x=2:y=4:w=20:h=30:qoffset=-0.6", chain)
        self.assertIn("x=40:y=50:w=60:h=70:qoffset=-0.6", chain)

    def test_encode_frame_uses_one_frame_and_all_rois(self):
        with mock.patch.object(encode_grid, "run") as run_mock:
            encode_grid.encode_frame(
                "input.mp4", 7, 640, 480,
                [(2, 4, 20, 30), (40, 50, 60, 70)],
                -0.3, 25.0, "/tmp", "test", 900.0,
            )

        cmd = run_mock.call_args.args[0]
        vf = cmd[cmd.index("-vf") + 1]
        self.assertEqual(vf.count(",addroi="), 2)
        self.assertEqual(cmd[cmd.index("-frames:v") + 1], "1")
        self.assertEqual(cmd[cmd.index("-g") + 1], "1")


class FrameGridTests(unittest.TestCase):
    def test_build_grid_schedules_every_combination_for_each_frame(self):
        metadata = {
            0: {"width": 100, "height": 100, "boxes": []},
            1: {"width": 100, "height": 100, "boxes": []},
        }

        def fake_worker(task):
            key = task[12]
            frame_idx = task[2]
            return key, frame_idx, {
                "bitrate_kbps": 500.0,
                "target_bitrate_kbps": 500.0,
                "vmaf": 0.0,
                "roi_vmaf": 0.0,
            }

        with mock.patch.object(encode_grid, "probe_video", return_value=(25.0, 100, 100, 2)), \
                mock.patch.object(encode_grid, "load_yolo_metadata", return_value=metadata), \
                mock.patch.object(encode_grid, "_grid_segment_worker", side_effect=fake_worker) as worker:
            result = encode_grid.build_grid(
                "input.mp4", "metadata.jsonl", metrics=(), workers=1,
                bitrate_levels_kbps=[500.0],
            )

        expected_jobs = 2 * len(RESOLUTIONS) * len(ROI_QOFFSET_LEVELS)
        self.assertEqual(worker.call_count, expected_jobs)
        self.assertEqual(result["grid_unit"], "frame")
        self.assertNotIn("segment_frames", result)
        self.assertNotIn("qp_levels", result)
        for curve in result["grid"].values():
            self.assertEqual(len(curve), 2)


class SegmentExportTests(unittest.TestCase):
    def test_segment_keeps_distinct_rois(self):
        action = EncoderAction(0.75, 0, 2)
        decisions = [
            (0, action, 1000.0, 2.0),
            (1, action, 1200.0, 2.0),
        ]
        metadata = {
            0: {"boxes": [[0, 0, 20, 20]]},
            1: {"boxes": [[40, 40, 60, 60]]},
        }
        env = mock.Mock(prev_bitrate=0.0)

        with mock.patch.object(
            export_encoded_video, "encode_segment_abr", return_value="segment.mp4"
        ) as encode_mock, mock.patch.object(
            export_encoded_video, "extract_bitrate_per_frame",
            return_value=[800.0, 900.0],
        ):
            segments = export_encoded_video.build_and_encode_segments(
                "input.mp4", decisions, metadata, 100, 100, 25.0, "/tmp", env,
            )

        passed_boxes = encode_mock.call_args.args[5]
        self.assertEqual(passed_boxes, [(0, 0, 128, 96), (256, 192, 128, 96)])
        self.assertEqual(segments[0]["roi_count"], 2)
        self.assertTrue(segments[0]["roi_applied"])


class StrictGridContractTests(unittest.TestCase):
    @staticmethod
    def trace(frame_idx=0):
        return [{
            "frame_idx": frame_idx,
            "semantic_score": 1.0,
            "bandwidth": 1000.0,
            "motion": 0.1,
            "roi_area": 0.2,
        }]

    @staticmethod
    def grid():
        curves = {}
        cell = {"bitrate_kbps": 1000.0, "vmaf": 80.0, "roi_vmaf": 82.0}
        for resolution_idx in range(len(RESOLUTIONS)):
            for roi_idx in range(len(ROI_QOFFSET_LEVELS)):
                curves[f"res{resolution_idx}_br0_roi{roi_idx}"] = [dict(cell)]
        return {
            "grid_unit": "frame",
            "num_frames": 1,
            "resolutions": [list(value) for value in RESOLUTIONS],
            "bitrate_levels_kbps": [1000.0],
            "roi_qoffset_levels": ROI_QOFFSET_LEVELS,
            "grid": curves,
        }

    def write_grid(self, value):
        handle = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(value, handle)
        handle.close()
        self.addCleanup(os.unlink, handle.name)
        return handle.name

    def test_legacy_grid_is_rejected_instead_of_falling_back(self):
        legacy = self.grid()
        del legacy["grid_unit"]

        with self.assertRaisesRegex(ValueError, "grid_unit"):
            VCUSimEnv(trace=self.trace(), encode_grid_path=self.write_grid(legacy))

    def test_frame_index_does_not_wrap_around(self):
        env = VCUSimEnv(
            trace=self.trace(frame_idx=1),
            encode_grid_path=self.write_grid(self.grid()),
        )
        env.reset()

        with self.assertRaisesRegex(IndexError, "frame_idx=1"):
            env.step(0)


if __name__ == "__main__":
    unittest.main()
