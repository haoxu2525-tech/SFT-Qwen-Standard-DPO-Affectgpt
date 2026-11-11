"""Network-free checks of retry, audit rejection, and streaming upload behavior."""
import ast
import io
import json
import os
from pathlib import Path
import random
import subprocess
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import urllib.error
from urllib.parse import urlsplit
import uuid
import base64


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        names = {"UploadError", "temporary_video", "save_json", "input_rejected", "annotate_one"}
        tree = ast.parse(Path(__file__).with_name("annotate_api.py").read_text(encoding="utf-8"))
        self.ns = dict(globals(), requests=SimpleNamespace(get=Mock()), NEGATIVE_MODEL="qwen3.8-omni-flash",
                       video_digest=lambda p: "unchanged", qwen_negative=Mock())
        exec(compile(ast.Module(body=[n for n in tree.body if getattr(n, "name", None) in names],
                               type_ignores=[]), "annotate_api.py", "exec"), self.ns)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root/"video.mp4").write_bytes(b"original bytes")
        self.row = dict(id="sample", video="video.mp4", prompt="original question", ground_truth_answer="chosen")
        self.good = dict(content="rejected", stream_complete=True, finish_reason="stop", usage={})

    def run_one(self):
        with patch.object(time, "sleep") as sleep:
            result = self.ns["annotate_one"](self.row, self.root, "https://api.example", "secret",
                {"max_tokens": None}, self.root, 3)
        self.assertEqual(json.loads((self.root/"sample.json").read_text()), result)
        self.assertEqual(result["chosen"], "chosen")
        return result, sleep

    def test_transient_failure_then_success(self):
        self.ns["qwen_negative"].side_effect = [ConnectionError(), self.good]
        r, sleep = self.run_one()
        self.assertEqual(r["rejected"], "rejected")
        self.assertEqual(len(r["annotation_api"]["attempts"]), 2)
        self.assertNotIn("error", r["annotation_api"])
        self.assertEqual(sleep.call_count, 1)

    def test_persistent_failure_is_saved_and_returns(self):
        self.ns["qwen_negative"].side_effect = ConnectionError()
        r, sleep = self.run_one()
        self.assertEqual(r["rejected"], "")
        self.assertEqual(len(r["annotation_api"]["attempts"]), 3)
        self.assertEqual(sleep.call_count, 2)

    def test_audit_rejection_is_not_retried(self):
        self.ns["qwen_negative"].side_effect = urllib.error.HTTPError("https://api.example", 400, "audit", {},
            io.BytesIO(b'{"code":"data_inspection_failed"}'))
        r, sleep = self.run_one()
        self.assertTrue(self.ns["input_rejected"](r))
        self.assertEqual(self.ns["qwen_negative"].call_count, 1)
        sleep.assert_not_called()

    def test_rate_limit_backs_off(self):
        self.ns["qwen_negative"].side_effect = [urllib.error.HTTPError("https://api.example", 429, "limit", {},
            io.BytesIO(b'limit')), self.good]
        r, sleep = self.run_one()
        self.assertEqual(r["rejected"], "rejected")
        self.assertGreaterEqual(sleep.call_args.args[0], 60)

    def test_upload_streams_and_hides_credentials_from_argv(self):
        policy = dict(upload_host="https://bucket.oss-cn-beijing.aliyuncs.com", upload_dir="private",
            oss_access_key_id="credential", signature="signature", policy="signed-policy",
            x_oss_object_acl="private", x_oss_forbid_overwrite="true")
        self.ns["requests"].get.return_value = SimpleNamespace(status_code=200, json=lambda: {"data": policy})
        with patch.object(subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="200")) as run:
            result = self.ns["temporary_video"](self.root/"video.mp4", "https://api.example/compatible-mode/v1", "secret")
        self.assertTrue(result.startswith("oss://private/"))
        self.assertNotIn("credential", str(run.call_args.args))
        self.assertIn('form-string = "OSSAccessKeyId=credential"', run.call_args.kwargs["input"])
        self.assertIn('form = "file=@', run.call_args.kwargs["input"])


if __name__ == "__main__":
    unittest.main()
