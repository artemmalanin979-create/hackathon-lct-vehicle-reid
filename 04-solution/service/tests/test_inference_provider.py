"""The requested accelerator must match the backend actually running ONNX."""

import ctypes
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from app.core import model


CUDA = "CUDAExecutionProvider"
CPU = "CPUExecutionProvider"


class _Session:
    def __init__(self, active_providers):
        self.active_providers = active_providers
        self.fallback_enabled = True

    def disable_fallback(self):
        self.fallback_enabled = False

    def get_providers(self):
        return self.active_providers

    def get_inputs(self):
        return [SimpleNamespace(name="images")]


class InferenceProviderTests(unittest.TestCase):
    def _embedder(self, mode, factory, driver_error=False):
        whitening = (np.eye(512, dtype=np.float32), np.zeros(512, dtype=np.float32))
        def set_device_count(pointer):
            ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int))[0] = 1
            return 0

        driver = SimpleNamespace(cuInit=Mock(return_value=0),
                                 cuDeviceGetCount=Mock(side_effect=set_device_count))
        driver_loader = Mock(side_effect=OSError("libcuda.so.1 missing")) if driver_error else Mock(return_value=driver)
        with patch.dict(os.environ, {"LCT_DEVICE": mode}), \
             patch.object(ctypes, "CDLL", driver_loader), \
             patch.object(model.ort, "get_available_providers", return_value=[CUDA, CPU]), \
             patch.object(model.ort, "InferenceSession", side_effect=factory), \
             patch.object(model, "_load_whitening", return_value=whitening):
            return model.Embedder(Path("osnet.onnx"), Path("combined.onnx"),
                                  Path("whitening.npz"), verify_sha256=False)

    def test_auto_prefers_cuda_for_both_onnx_models(self):
        requested = []
        sessions = []

        def open_session(*args, **kwargs):
            requested.append(kwargs["providers"])
            session = _Session([CUDA, CPU])
            sessions.append(session)
            return session

        embedder = self._embedder("auto", open_session)
        self.assertEqual(requested, [[CUDA, CPU], [CUDA, CPU]])
        self.assertTrue(all(not session.fallback_enabled for session in sessions))
        self.assertEqual(embedder.inference_backend, {
            "requested_device": "auto",
            "active_device": "cuda",
            "providers": {"osnet": [CUDA, CPU], "combined_v1": [CUDA, CPU]},
        })

    def test_explicit_cuda_rejects_ort_silent_cpu_fallback(self):
        with self.assertRaisesRegex(RuntimeError, "CUDA"):
            self._embedder("cuda", lambda *args, **kwargs: _Session([CPU]))

    def test_explicit_cuda_rejects_session_that_cannot_disable_runtime_fallback(self):
        class SessionWithoutFallbackControl:
            def get_providers(self):
                return [CUDA, CPU]

            def get_inputs(self):
                return [SimpleNamespace(name="images")]

        with self.assertRaisesRegex(RuntimeError, "CUDA"):
            self._embedder("cuda", lambda *args, **kwargs: SessionWithoutFallbackControl())

    def test_explicit_cpu_avoids_cuda_and_reports_cpu(self):
        requested = []

        def open_session(*args, **kwargs):
            requested.append(kwargs["providers"])
            return _Session([CPU])

        embedder = self._embedder("cpu", open_session)
        self.assertEqual(requested, [[CPU], [CPU]])
        self.assertEqual(embedder.inference_backend["active_device"], "cpu")
        self.assertEqual(embedder.inference_backend["providers"], {
            "osnet": [CPU], "combined_v1": [CPU],
        })

    def test_auto_reports_cpu_when_cuda_compiled_but_unavailable_at_runtime(self):
        requested = []

        def open_session(*args, **kwargs):
            requested.append(kwargs["providers"])
            return _Session([CPU])

        embedder = self._embedder("auto", open_session)
        self.assertEqual(requested[0], [CUDA, CPU])
        self.assertEqual(requested[-2:], [[CPU], [CPU]])
        self.assertEqual(embedder.inference_backend["active_device"], "cpu")

    def test_auto_never_constructs_cuda_session_without_visible_driver(self):
        requested = []

        def open_session(*args, **kwargs):
            requested.append(kwargs["providers"])
            return _Session([CPU])

        embedder = self._embedder("auto", open_session, driver_error=True)
        self.assertEqual(requested, [[CPU], [CPU]])
        self.assertEqual(embedder.inference_backend["active_device"], "cpu")

    def test_strict_cuda_fails_before_onnx_without_visible_driver(self):
        requested = []

        def open_session(*args, **kwargs):
            requested.append(kwargs["providers"])
            return _Session([CPU])

        with self.assertRaisesRegex(RuntimeError, "GPU|CUDA"):
            self._embedder("cuda", open_session, driver_error=True)
        self.assertEqual(requested, [])


if __name__ == "__main__":
    unittest.main()
