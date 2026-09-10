"""Unit tests for the inductor compile-backend track (stage1 design).

Covers (stage design/stage1/04_测试与验收.md §T1):
  1. track predicate ``_inductor_track_enabled``
  2. pass_key / get_pass_manager_cls switch (env carrier)
  3. early-hook derived defaults ``NPUPlatform._apply_inductor_track_defaults``
  4. late-hook env carriers ``NPUPlatform._setup_inductor_track_envs``
  5. ``_setup_compile_backend`` cg=NONE guard (track keeps VLLM_COMPILE)
  6. fail-fast on incompatible user options (enforce_eager / -O0 / standalone=1 / MEGA=1)

Config-level only: no NPU device is initialized.
"""

import os
from types import SimpleNamespace
from unittest.mock import patch

from vllm.config import CompilationConfig, CompilationMode, VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.config.vllm import OptimizationLevel

from tests.ut.base import TestBase
from vllm_ascend.utils import COMPILATION_PASS_KEY

# All env vars the track may touch; saved/restored around every test.
_TRACK_ENV_VARS = (
    "VLLM_ASCEND_COMPILE_BACKEND",
    "TORCHINDUCTOR_NPU_BACKEND",
    "VLLM_USE_STANDALONE_COMPILE",
    "VLLM_USE_AOT_COMPILE",
    "VLLM_USE_MEGA_AOT_ARTIFACT",
    "VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE",
    "VLLM_ENABLE_INDUCTOR_COORDINATE_DESCENT_TUNING",
)

# vLLM upstream defaults for pass_key / get_pass_manager_cls (platforms/interface.py).
_UPSTREAM_PASS_KEY = "post_grad_custom_post_pass"
_UPSTREAM_PASS_MANAGER = "vllm_ascend.compilation.ascend_post_grad_pass_manager.AscendPostGradPassManager"
_ASCEND_PASS_MANAGER = "vllm_ascend.compilation.graph_fusion_pass_manager.GraphFusionPassManager"


def _clear_track_envs() -> None:
    for name in _TRACK_ENV_VARS:
        os.environ.pop(name, None)


class TrackTestBase(TestBase):
    def setUp(self):
        super().setUp()
        self._saved_env = {name: os.environ.get(name) for name in _TRACK_ENV_VARS}
        _clear_track_envs()

    def tearDown(self):
        for name, value in self._saved_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        super().tearDown()

    @staticmethod
    def _make_vllm_config(compile_backend: str = "auto") -> VllmConfig:
        """Bare VllmConfig with the track switch in additional_config.

        check_and_update_config is patched out (see test_graph_fusion_pass_manager.py),
        so the late hook does not run here; the early hook sees additional_config
        during VllmConfig.__post_init__ only if device_type already resolves to npu.
        """
        with patch("vllm_ascend.platform.NPUPlatform.check_and_update_config"):
            vllm_config = VllmConfig()
        # A bare VllmConfig() has model_config=None; the early hook's guard
        # (mirroring check_and_update_config) skips such configs.
        vllm_config.model_config = SimpleNamespace(enforce_eager=False)
        vllm_config.additional_config = {"ascend_compilation_config": {"compile_backend": compile_backend}}
        vllm_config.device_config.device_type = "npu"
        return vllm_config


class TestInductorTrackPredicate(TrackTestBase):
    def test_disabled_by_default(self):
        from vllm_ascend.platform import _inductor_track_enabled

        self.assertFalse(_inductor_track_enabled(self._make_vllm_config("auto")))
        vllm_config = self._make_vllm_config("auto")
        vllm_config.additional_config = None
        self.assertFalse(_inductor_track_enabled(vllm_config))

    def test_enabled_when_compile_backend_is_inductor(self):
        from vllm_ascend.platform import _inductor_track_enabled

        self.assertTrue(_inductor_track_enabled(self._make_vllm_config("inductor")))


class TestPassKeySwitch(TrackTestBase):
    def test_default_keeps_ascend_pass_machinery(self):
        platform = __import__("vllm_ascend.platform", fromlist=["NPUPlatform"]).NPUPlatform()
        self.assertEqual(platform.pass_key, COMPILATION_PASS_KEY)
        self.assertEqual(platform.get_pass_manager_cls(), _ASCEND_PASS_MANAGER)

    def test_track_switches_to_upstream_pass_machinery(self):
        os.environ["VLLM_ASCEND_COMPILE_BACKEND"] = "inductor"
        platform = __import__("vllm_ascend.platform", fromlist=["NPUPlatform"]).NPUPlatform()
        self.assertEqual(platform.pass_key, _UPSTREAM_PASS_KEY)
        self.assertEqual(platform.get_pass_manager_cls(), _UPSTREAM_PASS_MANAGER)


class TestApplyInductorTrackDefaults(TrackTestBase):
    def test_track_off_is_a_noop(self):
        from vllm_ascend.platform import NPUPlatform

        vllm_config = self._make_vllm_config("auto")
        backend_before = vllm_config.compilation_config.backend
        NPUPlatform._apply_inductor_track_defaults(vllm_config)
        cc = vllm_config.compilation_config
        self.assertNotEqual(cc.backend, "inductor")
        self.assertEqual(cc.backend, backend_before)
        self.assertNotEqual(cc.cudagraph_mode, CUDAGraphMode.NONE)
        self.assertTrue(cc.inductor_compile_config.get("combo_kernels", True))

    def test_track_on_derives_inductor_semantics(self):
        from vllm_ascend.platform import NPUPlatform, _INDUCTOR_TRACK_PASS_FLAGS_OFF

        vllm_config = self._make_vllm_config("inductor")
        NPUPlatform._apply_inductor_track_defaults(vllm_config)
        cc = vllm_config.compilation_config
        self.assertEqual(cc.backend, "inductor")
        self.assertEqual(cc.cudagraph_mode, CUDAGraphMode.NONE)
        self.assertFalse(cc.ir_enable_torch_wrap)
        self.assertFalse(cc.inductor_compile_config["combo_kernels"])
        self.assertFalse(cc.inductor_compile_config["benchmark_combo_kernel"])
        for flag in _INDUCTOR_TRACK_PASS_FLAGS_OFF:
            self.assertFalse(getattr(cc.pass_config, flag), flag)

    def test_track_on_core_derives_custom_ops_none(self):
        """Full-chain: with the switch present at VllmConfig construction time,
        the early hook rewrites backend before vLLM core derives custom_ops."""
        with patch(
            "vllm_ascend.platform.NPUPlatform.check_and_update_config"
        ), patch(
            "vllm_ascend.platform._get_default_max_cudagraph_capture_size",
            return_value=None,
        ):
            vllm_config = VllmConfig(
                additional_config={"ascend_compilation_config": {"compile_backend": "inductor"}},
            )
        if vllm_config.device_config.device_type != "npu":
            self.skipTest("current_platform did not resolve to npu")
        cc = vllm_config.compilation_config
        self.assertEqual(cc.backend, "inductor")
        self.assertIn("none", cc.custom_ops)
        self.assertNotIn("all", cc.custom_ops)
        self.assertEqual(cc.mode, CompilationMode.VLLM_COMPILE)
        self.assertEqual(cc.cudagraph_mode, CUDAGraphMode.NONE)

    def test_track_on_rejects_enforce_eager(self):
        from vllm_ascend.platform import NPUPlatform

        vllm_config = self._make_vllm_config("inductor")
        vllm_config.model_config.enforce_eager = True
        with self.assertRaises(ValueError):
            NPUPlatform._apply_inductor_track_defaults(vllm_config)

    def test_track_on_rejects_o0(self):
        from vllm_ascend.platform import NPUPlatform

        vllm_config = self._make_vllm_config("inductor")
        vllm_config.optimization_level = OptimizationLevel.O0
        with self.assertRaises(ValueError):
            NPUPlatform._apply_inductor_track_defaults(vllm_config)

    def test_non_npu_device_is_skipped(self):
        from vllm_ascend.platform import NPUPlatform

        vllm_config = self._make_vllm_config("inductor")
        vllm_config.device_config.device_type = "cuda"
        NPUPlatform._apply_inductor_track_defaults(vllm_config)
        self.assertNotEqual(vllm_config.compilation_config.backend, "inductor")


class TestSetupInductorTrackEnvs(TrackTestBase):
    @staticmethod
    def _make_stubs(compile_backend: str = "inductor", backend: str | None = None):
        compilation_config = CompilationConfig() if backend is None else CompilationConfig(backend=backend)
        vllm_config = SimpleNamespace(compilation_config=compilation_config)
        ascend_config = SimpleNamespace(
            ascend_compilation_config=SimpleNamespace(compile_backend=compile_backend)
        )
        return vllm_config, ascend_config

    def test_track_off_writes_no_env(self):
        from vllm_ascend.platform import NPUPlatform

        vllm_config, ascend_config = self._make_stubs(compile_backend="auto")
        NPUPlatform._setup_inductor_track_envs(vllm_config, ascend_config)
        for name in _TRACK_ENV_VARS:
            self.assertNotIn(name, os.environ, name)

    def test_track_on_sets_all_env_carriers(self):
        from vllm_ascend.platform import NPUPlatform

        vllm_config, ascend_config = self._make_stubs()
        NPUPlatform._setup_inductor_track_envs(vllm_config, ascend_config)
        self.assertEqual(os.environ["TORCHINDUCTOR_NPU_BACKEND"], "triton_experimental")
        self.assertEqual(os.environ["VLLM_ASCEND_COMPILE_BACKEND"], "inductor")
        self.assertEqual(os.environ["VLLM_USE_STANDALONE_COMPILE"], "0")
        self.assertEqual(os.environ["VLLM_USE_AOT_COMPILE"], "0")
        self.assertEqual(os.environ["VLLM_USE_MEGA_AOT_ARTIFACT"], "0")
        self.assertEqual(os.environ["VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE"], "0")
        self.assertEqual(os.environ["VLLM_ENABLE_INDUCTOR_COORDINATE_DESCENT_TUNING"], "0")
        # backend is corrected to "inductor" even if the early hook was skipped
        self.assertEqual(vllm_config.compilation_config.backend, "inductor")

    def test_user_npu_backend_is_respected(self):
        from vllm_ascend.platform import NPUPlatform

        os.environ["TORCHINDUCTOR_NPU_BACKEND"] = "default"
        vllm_config, ascend_config = self._make_stubs()
        NPUPlatform._setup_inductor_track_envs(vllm_config, ascend_config)
        self.assertEqual(os.environ["TORCHINDUCTOR_NPU_BACKEND"], "default")

    def test_track_on_rejects_standalone_1(self):
        from vllm_ascend.platform import NPUPlatform

        os.environ["VLLM_USE_STANDALONE_COMPILE"] = "1"
        vllm_config, ascend_config = self._make_stubs()
        with self.assertRaises(ValueError):
            NPUPlatform._setup_inductor_track_envs(vllm_config, ascend_config)

    def test_track_on_rejects_mega_aot_1(self):
        from vllm_ascend.platform import NPUPlatform

        os.environ["VLLM_USE_MEGA_AOT_ARTIFACT"] = "1"
        vllm_config, ascend_config = self._make_stubs()
        with self.assertRaises(ValueError):
            NPUPlatform._setup_inductor_track_envs(vllm_config, ascend_config)

    def test_user_autotune_1_is_kept(self):
        from vllm_ascend.platform import NPUPlatform

        os.environ["VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE"] = "1"
        vllm_config, ascend_config = self._make_stubs()
        NPUPlatform._setup_inductor_track_envs(vllm_config, ascend_config)
        self.assertEqual(os.environ["VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE"], "1")

    def test_user_aot_compile_1_is_kept(self):
        from vllm_ascend.platform import NPUPlatform

        os.environ["VLLM_USE_AOT_COMPILE"] = "1"
        vllm_config, ascend_config = self._make_stubs()
        NPUPlatform._setup_inductor_track_envs(vllm_config, ascend_config)
        self.assertEqual(os.environ["VLLM_USE_AOT_COMPILE"], "1")
        # MEGA is still pinned off: it requires the standalone path.
        self.assertEqual(os.environ["VLLM_USE_MEGA_AOT_ARTIFACT"], "0")


class TestAscendPostGradPassManager(TrackTestBase):
    def test_fix_functionalization_replaced_with_noop(self):
        import torch.fx

        from vllm_ascend.compilation.ascend_post_grad_pass_manager import (
            AscendNoopInductorPass,
            AscendPostGradPassManager,
        )

        with patch(
            "vllm_ascend.platform.NPUPlatform.check_and_update_config"
        ), patch(
            "vllm_ascend.platform._get_default_max_cudagraph_capture_size",
            return_value=None,
        ):
            # Build through the track so the early hook pins the fusion flags
            # off; a bare config leaves them on and PostGradPassManager.configure
            # then references CUDA-gated fusion classes (R2 NameError).
            vllm_config = VllmConfig(
                additional_config={"ascend_compilation_config": {"compile_backend": "inductor"}},
            )
        if vllm_config.device_config.device_type != "npu":
            self.skipTest("current_platform did not resolve to npu")
        manager = AscendPostGradPassManager()
        manager.configure(vllm_config)
        # FixFunctionalizationPass.__call__ references CUDA-only ops
        # (torch.ops._C.rotary_embedding) and must be replaced on NPU.
        self.assertIsInstance(manager.fix_functionalization, AscendNoopInductorPass)
        # The replacement keeps a working uuid (inductor code cache keys).
        # The remaining always-on chain (post_cleanup / ir_lowering /
        # clone_elimination) needs a real PassContext and is exercised
        # end-to-end by the T2 smoke run instead.
        self.assertTrue(manager.fix_functionalization.uuid())


class TestSetupCompileBackendGuard(TrackTestBase):
    @staticmethod
    def _make_stub(compile_backend: str) -> SimpleNamespace:
        compilation_config = CompilationConfig()
        compilation_config.mode = CompilationMode.VLLM_COMPILE
        compilation_config.cudagraph_mode = CUDAGraphMode.NONE
        return SimpleNamespace(
            compilation_config=compilation_config,
            additional_config={
                "ascend_compilation_config": {
                    "compile_backend": compile_backend,
                    "enable_npugraph_ex": True,
                    "enable_static_kernel": False,
                }
            },
            model_config=SimpleNamespace(enforce_eager=False),
            parallel_config=SimpleNamespace(
                all2all_backend="flashinfer_all2allv",
                tensor_parallel_size=1,
                data_parallel_size=1,
            ),
            _set_cudagraph_sizes=lambda: None,
        )

    def _run(self, compile_backend: str):
        from vllm_ascend.platform import _setup_compile_backend

        vllm_config = self._make_stub(compile_backend)
        with patch("vllm_ascend.platform.enable_sp", return_value=False):
            _setup_compile_backend(
                vllm_config,
                compile_backend="vllm_ascend.compilation.compiler_interface.AscendCompiler",
            )
        return vllm_config

    def test_track_off_keeps_current_behavior_mode_forced_none(self):
        vllm_config = self._run("auto")
        self.assertEqual(vllm_config.compilation_config.mode, CompilationMode.NONE)
        self.assertFalse(vllm_config.additional_config["ascend_compilation_config"]["enable_npugraph_ex"])

    def test_track_on_keeps_vllm_compile(self):
        vllm_config = self._run("inductor")
        self.assertEqual(vllm_config.compilation_config.mode, CompilationMode.VLLM_COMPILE)
        self.assertFalse(vllm_config.additional_config["ascend_compilation_config"]["enable_npugraph_ex"])
        self.assertFalse(vllm_config.additional_config["ascend_compilation_config"]["enable_static_kernel"])
