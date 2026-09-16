"""Unit tests for the inductor compile-backend track (stage1 design).

Covers (stage design/stage1/04_测试与验收.md §T1 + stage2 rev B):
  1. track predicate ``_inductor_track_enabled``
  2. pass_key / get_pass_manager_cls switch (AscendConfig singleton — per-engine
     scoped, no env carrier to leak or clean up)
  3. early-hook derived defaults ``NPUPlatform._apply_inductor_track_defaults``
  4. late-hook env setup ``NPUPlatform._setup_inductor_track_envs``
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
    "TORCHINDUCTOR_NPU_BACKEND",
    "VLLM_USE_STANDALONE_COMPILE",
    "VLLM_USE_AOT_COMPILE",
    "VLLM_USE_MEGA_AOT_ARTIFACT",
    "VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE",
    "VLLM_ENABLE_INDUCTOR_COORDINATE_DESCENT_TUNING",
    "VLLM_USE_BREAKABLE_CUDAGRAPH",
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
    """Stage2 rev B: the pass switch reads the AscendConfig singleton
    (refreshed per engine by init_ascend_config) instead of an env carrier —
    per-engine scoping with nothing to leak or clean up."""

    def setUp(self):
        super().setUp()
        from vllm_ascend.ascend_config import clear_ascend_config

        clear_ascend_config()

    def tearDown(self):
        from vllm_ascend.ascend_config import clear_ascend_config

        clear_ascend_config()
        super().tearDown()

    def _init_engine(self, compile_backend: str = "auto"):
        from vllm_ascend.ascend_config import init_ascend_config

        vllm_config = self._make_vllm_config(compile_backend)
        init_ascend_config(vllm_config)

    def test_uninitialized_keeps_ascend_pass_machinery(self):
        platform = __import__("vllm_ascend.platform", fromlist=["NPUPlatform"]).NPUPlatform()
        self.assertEqual(platform.pass_key, COMPILATION_PASS_KEY)
        self.assertEqual(platform.get_pass_manager_cls(), _ASCEND_PASS_MANAGER)

    def test_track_engine_switches_to_upstream_pass_machinery(self):
        self._init_engine("inductor")
        platform = __import__("vllm_ascend.platform", fromlist=["NPUPlatform"]).NPUPlatform()
        self.assertEqual(platform.pass_key, _UPSTREAM_PASS_KEY)
        self.assertEqual(platform.get_pass_manager_cls(), _UPSTREAM_PASS_MANAGER)

    def test_track_off_engine_reverts_after_track_on_engine(self):
        """Red line (总纲 §〇): a track-off engine after a track-on engine in
        the same process must observe the legacy pass machinery — guaranteed
        by the singleton refresh, no cleanup needed."""
        self._init_engine("inductor")
        self._init_engine("auto")
        platform = __import__("vllm_ascend.platform", fromlist=["NPUPlatform"]).NPUPlatform()
        self.assertEqual(platform.pass_key, COMPILATION_PASS_KEY)
        self.assertEqual(platform.get_pass_manager_cls(), _ASCEND_PASS_MANAGER)


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

    def test_track_on_leaves_cudagraph_mode_to_preset(self):
        """Debt 2 (ledger 13): the early hook no longer fills a cudagraph_mode
        default — None stays None so the -O presets own it (O1 -> PIECEWISE,
        O2/O3 -> FULL_AND_PIECEWISE, upstream semantics)."""
        from vllm_ascend.platform import NPUPlatform, _INDUCTOR_TRACK_PASS_FLAGS_OFF

        vllm_config = self._make_vllm_config("inductor")
        # Early-hook timing: -O presets have not filled cudagraph_mode yet.
        vllm_config.compilation_config.cudagraph_mode = None
        NPUPlatform._apply_inductor_track_defaults(vllm_config)
        cc = vllm_config.compilation_config
        self.assertEqual(cc.backend, "inductor")
        self.assertIsNone(cc.cudagraph_mode)
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
        # Debt 2: default -O2 journey — the presets now own the track default.
        self.assertEqual(cc.cudagraph_mode, CUDAGraphMode.FULL_AND_PIECEWISE)

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

    def test_track_on_warns_when_breakable_wins(self):
        """Debt 1 (ledger 13): upstream semantics — breakable wins, the track is
        inert (mode already forced to NONE upstream). No raise; a dedicated
        warning carries the escape hatch ("set it to 0": unsetting would
        re-trigger the nine-architecture auto-inject). The hook continues, so
        the env carriers are still written."""
        import logging

        from vllm_ascend.platform import NPUPlatform

        os.environ["VLLM_USE_BREAKABLE_CUDAGRAPH"] = "1"
        vllm_config, ascend_config = self._make_stubs()
        with self.assertLogs("vllm", level=logging.WARNING) as logs:
            NPUPlatform._setup_inductor_track_envs(vllm_config, ascend_config)
        joined = "\n".join(logs.output)
        self.assertIn("VLLM_USE_BREAKABLE_CUDAGRAPH wins", joined)
        self.assertIn("inductor track is inert", joined)
        self.assertIn("Set VLLM_USE_BREAKABLE_CUDAGRAPH=0", joined)
        # the hook ran to completion (no raise): env carriers still written
        self.assertEqual(os.environ.get("TORCHINDUCTOR_NPU_BACKEND"), "triton_experimental")

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


class TestCompileBackendEnum(TrackTestBase):
    """Stage2: compile_backend four-value enum + enable_npugraph_ex sentinel."""

    def _config(self, **kwargs):
        from vllm_ascend.ascend_config import AscendCompilationConfig

        return AscendCompilationConfig(**kwargs)

    def test_auto_default_resolves_npugraph_ex_true(self):
        config = self._config()
        self.assertEqual(config.compile_backend, "auto")
        self.assertTrue(config.enable_npugraph_ex)

    def test_inductor_resolves_false_and_keeps_minimal_usage(self):
        config = self._config(compile_backend="inductor")
        self.assertFalse(config.enable_npugraph_ex)

    def test_inductor_with_explicit_npugraph_ex_raises(self):
        with self.assertRaises(ValueError):
            self._config(compile_backend="inductor", enable_npugraph_ex=True)

    def test_fusion_pass_resolves_false(self):
        self.assertFalse(self._config(compile_backend="fusion_pass").enable_npugraph_ex)

    def test_npugraph_ex_resolves_true(self):
        self.assertTrue(self._config(compile_backend="npugraph_ex").enable_npugraph_ex)

    def test_auto_keeps_explicit_bool(self):
        self.assertFalse(self._config(enable_npugraph_ex=False).enable_npugraph_ex)
        self.assertTrue(self._config(enable_npugraph_ex=True).enable_npugraph_ex)

    def test_invalid_value_rejected(self):
        with self.assertRaises(ValueError):
            self._config(compile_backend="bogus")

    def test_reparse_idempotent(self):
        # additional_config writeback carries the resolved bool; re-parsing
        # (as workers do) must not trip the conflict check again.
        config = self._config(compile_backend="inductor", enable_npugraph_ex=False)
        self.assertFalse(config.enable_npugraph_ex)


class TestTrackCudagraphMode(TrackTestBase):
    """Debt 2 (ledger 13): the track default follows the -O presets
    (O1 -> PIECEWISE, O2/O3 -> FULL_AND_PIECEWISE); explicit values honored."""

    def test_default_follows_optimization_level_preset(self):
        from vllm_ascend.platform import NPUPlatform

        expected = {
            OptimizationLevel.O1: CUDAGraphMode.PIECEWISE,
            OptimizationLevel.O2: CUDAGraphMode.FULL_AND_PIECEWISE,
            OptimizationLevel.O3: CUDAGraphMode.FULL_AND_PIECEWISE,
        }
        for level, want in expected.items():
            with self.subTest(level=level):
                with patch(
                    "vllm_ascend.platform.NPUPlatform.check_and_update_config"
                ), patch(
                    "vllm_ascend.platform._get_default_max_cudagraph_capture_size",
                    return_value=None,
                ):
                    vllm_config = VllmConfig(
                        optimization_level=level,
                        additional_config={"ascend_compilation_config": {"compile_backend": "inductor"}},
                    )
                if vllm_config.device_config.device_type != "npu":
                    self.skipTest("current_platform did not resolve to npu")
                self.assertEqual(vllm_config.compilation_config.cudagraph_mode, want)

    def test_explicit_none_kept(self):
        from vllm_ascend.platform import NPUPlatform

        vllm_config = self._make_vllm_config("inductor")
        vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
        NPUPlatform._apply_inductor_track_defaults(vllm_config)
        self.assertEqual(vllm_config.compilation_config.cudagraph_mode, CUDAGraphMode.NONE)

    def test_full_family_modes_accepted(self):
        """Stage3 U6（三值全开）：FULL 族三值早 hook 放行，值原样保留、backend=inductor。

        迁移自 stage2 ``test_full_modes_raise``（契约迁移纪律 04 §T3-2：删除的只是
        错误文案中自声明「stage-3 支持」的临时约束，换等强度正向断言）。
        """
        from vllm_ascend.platform import NPUPlatform

        for mode in (
            CUDAGraphMode.FULL,
            CUDAGraphMode.FULL_AND_PIECEWISE,
            CUDAGraphMode.FULL_DECODE_ONLY,
        ):
            with self.subTest(mode=mode):
                vllm_config = self._make_vllm_config("inductor")
                vllm_config.compilation_config.cudagraph_mode = mode
                NPUPlatform._apply_inductor_track_defaults(vllm_config)
                self.assertEqual(vllm_config.compilation_config.cudagraph_mode, mode)
                self.assertEqual(vllm_config.compilation_config.backend, "inductor")

    def test_explicit_full_family_survives_o2_preset(self):
        """R3-13：显式 FULL 族 cg 在 -O2 preset 下存活（preset 只填 None 字段）。"""
        for mode in (
            CUDAGraphMode.FULL,
            CUDAGraphMode.FULL_AND_PIECEWISE,
            CUDAGraphMode.FULL_DECODE_ONLY,
        ):
            with self.subTest(mode=mode):
                with patch(
                    "vllm_ascend.platform.NPUPlatform.check_and_update_config"
                ), patch(
                    "vllm_ascend.platform._get_default_max_cudagraph_capture_size",
                    return_value=None,
                ):
                    vllm_config = VllmConfig(
                        optimization_level=OptimizationLevel.O2,
                        compilation_config=CompilationConfig(cudagraph_mode=mode),
                        additional_config={"ascend_compilation_config": {"compile_backend": "inductor"}},
                    )
                if vllm_config.device_config.device_type != "npu":
                    self.skipTest("current_platform did not resolve to npu")
                self.assertEqual(vllm_config.compilation_config.cudagraph_mode, mode)
                self.assertEqual(vllm_config.compilation_config.backend, "inductor")


class TestTrackFullFamilyBranches(TrackTestBase):
    """Stage3 U6：三 cg 值在晚 hook `_setup_compile_backend` 的分支走位。

    - FULL_AND_PIECEWISE → requires_piecewise_compilation()=True → PIECEWISE 分支
      （splitting_ops 填充含 mla/dsa，npugraph_ex 关）——与 stage-2 已验收形态同构；
    - FULL / FULL_DECODE_ONLY → has_full_cudagraphs() 分支 → splitting_ops=[]
      （上游对齐形态：单图整编译，D3-4 修订；T0''-1 纯 FULL/FDO 探针已实证）。
    """

    @staticmethod
    def _make_stub(cg) -> SimpleNamespace:
        compilation_config = CompilationConfig()
        compilation_config.mode = CompilationMode.VLLM_COMPILE
        compilation_config.cudagraph_mode = cg
        return SimpleNamespace(
            compilation_config=compilation_config,
            additional_config={
                "ascend_compilation_config": {
                    "compile_backend": "inductor",
                    # step-6 (_update_compilation_modes) 已把解析后的 False 同步进来
                    "enable_npugraph_ex": False,
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

    def _run(self, cg):
        from vllm_ascend.platform import _setup_compile_backend

        vllm_config = self._make_stub(cg)
        with patch("vllm_ascend.platform.enable_sp", return_value=False):
            _setup_compile_backend(
                vllm_config,
                compile_backend="vllm_ascend.compilation.compiler_interface.AscendCompiler",
            )
        return vllm_config

    def test_full_and_piecewise_takes_piecewise_branch(self):
        vllm_config = self._run(CUDAGraphMode.FULL_AND_PIECEWISE)
        cc = vllm_config.compilation_config
        self.assertTrue(cc.splitting_ops, "PIECEWISE 分支应填充 splitting_ops")
        self.assertIn("vllm::mla_forward", cc.splitting_ops)
        self.assertIn("vllm::dsa_forward", cc.splitting_ops)
        self.assertFalse(vllm_config.additional_config["ascend_compilation_config"]["enable_npugraph_ex"])
        self.assertEqual(cc.cudagraph_mode, CUDAGraphMode.FULL_AND_PIECEWISE)

    def test_full_and_full_decode_only_take_upstream_aligned_branch(self):
        for mode in (CUDAGraphMode.FULL, CUDAGraphMode.FULL_DECODE_ONLY):
            with self.subTest(mode=mode):
                vllm_config = self._run(mode)
                cc = vllm_config.compilation_config
                self.assertEqual(cc.splitting_ops, [], "上游对齐形态应清空 splitting_ops（单图整编译）")
                self.assertEqual(cc.cudagraph_mode, mode)
                # inductor 轨上该分支不写 npugraph_ex（step-6 已同步 False，保持不变）
                self.assertFalse(vllm_config.additional_config["ascend_compilation_config"]["enable_npugraph_ex"])


class TestNpugraphExTrackGuard(TrackTestBase):
    """Stage2: explicit compile_backend='npugraph_ex' needs full-graph modes."""

    @staticmethod
    def _make_stub(compile_backend: str, cg) -> SimpleNamespace:
        compilation_config = CompilationConfig()
        compilation_config.mode = CompilationMode.VLLM_COMPILE
        compilation_config.cudagraph_mode = cg
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

    def _run(self, compile_backend: str, cg):
        from vllm_ascend.platform import _setup_compile_backend

        vllm_config = self._make_stub(compile_backend, cg)
        with patch("vllm_ascend.platform.enable_sp", return_value=False):
            _setup_compile_backend(
                vllm_config,
                compile_backend="vllm_ascend.compilation.compiler_interface.AscendCompiler",
            )
        return vllm_config

    def test_npugraph_ex_with_cg_none_raises(self):
        with self.assertRaises(ValueError):
            self._run("npugraph_ex", CUDAGraphMode.NONE)

    def test_npugraph_ex_with_cg_full_ok(self):
        vllm_config = self._run("npugraph_ex", CUDAGraphMode.FULL)
        self.assertEqual(vllm_config.compilation_config.cudagraph_mode, CUDAGraphMode.FULL)

    def test_auto_with_cg_none_unchanged(self):
        vllm_config = self._run("auto", CUDAGraphMode.NONE)
        self.assertEqual(vllm_config.compilation_config.mode, CompilationMode.NONE)


class TestRngWarn(TrackTestBase):
    """Stage2: bernoulli-family RNG under graph capture warns (03 R2')."""

    def test_warns_on_bernoulli(self):
        import torch.fx

        from vllm_ascend.compilation.ascend_post_grad_pass_manager import _warn_if_graph_contains_rng

        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        graph.output(graph.call_function(torch.ops.aten.bernoulli.default, (x,)))
        with self.assertLogs(
            "vllm_ascend.compilation.ascend_post_grad_pass_manager", level="WARNING"
        ):
            _warn_if_graph_contains_rng(graph)

    def test_no_warn_without_rng(self):
        import torch.fx

        from vllm_ascend.compilation.ascend_post_grad_pass_manager import _warn_if_graph_contains_rng

        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        graph.output(graph.call_function(torch.ops.aten.add.Tensor, (x, x)))
        with self.assertNoLogs(
            "vllm_ascend.compilation.ascend_post_grad_pass_manager", level="WARNING"
        ):
            _warn_if_graph_contains_rng(graph)


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


class TestBreakableArchAutoInject(TrackTestBase):
    """Debt 1 (ledger 13): with the opt-in pin removed, upstream's architecture
    auto-inject (vllm.py:1211-1234) must fire again on Ascend — a VllmConfig
    for one of the nine breakable architectures with the env var absent gets
    it auto-set to "1" and compilation mode forced to NONE (vllm.py:1236-1241)."""

    def test_pin_removed_at_import_time(self):
        """The stage1 pin (setdefault "0" at vllm_ascend.platform import) is
        gone: importing the platform must not create the env var."""
        import vllm_ascend.platform  # noqa: F401

        self.assertNotIn("VLLM_USE_BREAKABLE_CUDAGRAPH", os.environ)

    def test_nine_arch_config_auto_enables_breakable(self):
        import torch

        # SimpleNamespace stands in for ModelConfig: __post_init__ reads a
        # handful of attributes/methods around the inject branch; a real
        # ModelConfig would need an HF checkpoint. VllmConfig is a pydantic
        # dataclass (replace() would validate), so follow the established
        # pattern: construct bare, assign model_config directly, then re-run
        # __post_init__ — the inject branch lives there. Track OFF (no
        # additional_config).
        model_config = SimpleNamespace(
            architectures=["MiniMaxM3SparseForCausalLM"],
            architecture="MiniMaxM3SparseForCausalLM",
            # skip try_verify_and_update_config's registry/HF resolution
            config_updated=True,
            enforce_eager=False,
            dtype=torch.bfloat16,
            pooler_config=None,
            is_encoder_decoder=False,
            hf_config=SimpleNamespace(hidden_size=1024, is_encoder_decoder=False),
            get_hidden_size=lambda: 1024,
            verify_with_parallel_config=lambda *a, **k: None,
            verify_dual_chunk_attention_config=lambda *a, **k: None,
            is_moe=False,
            enable_return_routed_experts=False,
            quantization=None,
            multimodal_config=None,
            attention_chunk_size=None,
        )
        with patch(
            "vllm_ascend.platform.NPUPlatform.check_and_update_config"
        ), patch(
            "vllm_ascend.platform._get_default_max_cudagraph_capture_size",
            return_value=None,
        ), patch.object(
            VllmConfig, "_set_cudagraph_sizes", lambda self: None
        ), patch.object(
            VllmConfig, "_set_max_num_scheduled_tokens", lambda self: None
        ):
            vllm_config = VllmConfig()
            vllm_config.model_config = model_config
            vllm_config.__post_init__()
        self.assertEqual(os.environ.get("VLLM_USE_BREAKABLE_CUDAGRAPH"), "1")
        self.assertEqual(vllm_config.compilation_config.mode, CompilationMode.NONE)
