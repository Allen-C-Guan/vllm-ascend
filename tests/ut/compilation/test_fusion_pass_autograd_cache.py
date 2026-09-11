#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""T3-1/V7（stage3 项3）：fusion_pass 默认轨 AOTAutogradCache 修复的栈桩 UT。

torch>=2.10 上 vLLM 默认走 AOT 编译，其 autograd-cache 保存路径要求内层编译器返回
inductor OutputCode，而 fusion_pass 的 compile_inner 返回 GraphModule——唯一结局是
`unwrap_output_code` 断言崩溃（torch/_functorch/_aot_autograd/autograd_cache.py:1431，
存量问题，基线 9a76a2372 同崩）。修复 = fusion_pass_compile 内 per-track 关闭
enable_autograd_cache/enable_remote_autograd_cache（上游 InductorAdaptor 同款先例，
SP vllm/compilation/compiler_interface.py:609-618）。
"""
from unittest.mock import patch

import torch
import torch.fx

from tests.ut.base import TestBase
from vllm_ascend.utils import COMPILATION_PASS_KEY


def _tiny_graph() -> torch.fx.GraphModule:
    class M(torch.nn.Module):
        def forward(self, x, y):
            return x + y

    return torch.fx.symbolic_trace(M())


class TestFusionPassAutogradCacheOff(TestBase):
    def _run_fusion_pass_compile_with_stub(self) -> dict:
        """以栈桩 compile_fx 捕获调用期 functorch 两 flag 的取值。"""
        import vllm_ascend.compilation.compiler_interface as ci

        captured: dict = {}

        def fake_compile_fx(graph, example_inputs, inner_compile=None, decompositions=None, **kw):
            fc = torch._functorch.config
            captured["local"] = bool(fc.enable_autograd_cache)
            captured["remote"] = bool(fc.enable_remote_autograd_cache)
            return lambda *a, **k: None

        graph = _tiny_graph()
        compiler_config = {COMPILATION_PASS_KEY: lambda g: g}
        with patch.object(ci, "compile_fx", fake_compile_fx):
            compiled, handle = ci.fusion_pass_compile(
                graph=graph,
                example_inputs=[],
                compiler_config=compiler_config,
                compile_range=None,
            )
        captured["compiled"] = compiled is not None
        captured["handle"] = handle
        return captured

    def test_flags_disabled_during_compile_fx(self):
        """RED：修复前调用期两 flag 为默认 True；修复后（per-track patch）应为 False。"""
        captured = self._run_fusion_pass_compile_with_stub()
        self.assertFalse(
            captured["local"],
            "enable_autograd_cache must be False inside fusion_pass_compile's compile_fx",
        )
        self.assertFalse(
            captured["remote"],
            "enable_remote_autograd_cache must be False inside fusion_pass_compile's compile_fx",
        )

    def test_flags_restored_after_call(self):
        """patch 退出后恢复原值（线程级+栈帧级作用域，不外溢）。"""
        before_local = torch._functorch.config.enable_autograd_cache
        before_remote = torch._functorch.config.enable_remote_autograd_cache
        captured = self._run_fusion_pass_compile_with_stub()
        self.assertTrue(captured["compiled"])
        self.assertIsNone(captured["handle"])  # 该轨从未写过 vLLM 自有缓存句柄
        self.assertEqual(torch._functorch.config.enable_autograd_cache, before_local)
        self.assertEqual(torch._functorch.config.enable_remote_autograd_cache, before_remote)

    def test_scoped_even_when_user_forces_on(self):
        """用户 env 强制开启（TORCHINDUCTOR_AUTOGRAD_CACHE 语义）时同样在栈内压下。"""
        with torch._functorch.config.patch(
            {"enable_autograd_cache": True, "enable_remote_autograd_cache": True}
        ):
            captured = self._run_fusion_pass_compile_with_stub()
        self.assertFalse(captured["local"])
        self.assertFalse(captured["remote"])
        # 外层 patch 域内仍为 True（未被破坏）
        self.assertTrue(torch._functorch.config.enable_autograd_cache)
