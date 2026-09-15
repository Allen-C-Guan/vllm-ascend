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
"""Stage4 W2 (D1) UT: the bias=None dynamic-quant fusion patterns, CPU fx level.

Covers the two patterns added to norm_quant_fusion_pass.py against synthetic FX
graphs mirroring the pass-visible (post-AOT) forms of the 0.6B W8A8_DYNAMIC
graph (T0b-1R):

* ``AddRMSNormDynamicQuantPatternWithNoneBias`` — direct
  ``_C_ascend.npu_add_rms_norm_bias(x, residual, weight, None, eps)`` ->
  ``npu_dynamic_quant`` (the 0.6B dense 2D form: no view on this chain);
* ``AddRMSNormDynamicQuantPatternWithNoneBiasView`` — same node with a
  swallowed ``aten.view``/``aten.reshape`` leading-dims merge before the quant
  (3D-input form; the swallowed shape is restored by the replacement).

No NPU is executed: graphs are fake-tensor traced on CPU (meta kernels only),
matching is pure FX. The graph builder mirrors the real post-AOT shapes:
schema-default args dropped (``None``/``eps=1e-6``), ``dst_type=torch.int8``
recorded as ``dst_type=1``, and no materialized dead tuple getitems (removed
via eliminate_dead_code — fake tracing materializes them, AOT does not).
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch._inductor.pattern_matcher as pm
import torch.fx.experimental.proxy_tensor as proxy_tensor
from torch._subclasses.fake_tensor import FakeTensorMode

from tests.ut.base import TestBase
from vllm_ascend.utils import enable_custom_op

try:
    _CUSTOM_OPS_READY = enable_custom_op()
except Exception:
    _CUSTOM_OPS_READY = False

pytestmark = [
    pytest.mark.skipif(
        not _CUSTOM_OPS_READY,
        reason="_C_ascend custom ops unavailable: the bias=None dynamic-quant "
        "patterns only register on the enable_custom_op() track",
    ),
]

_FUSED_OP = torch.ops.npu.npu_add_rms_norm_dynamic_quant.default
_NORM_OP = torch.ops._C_ascend.npu_add_rms_norm_bias.default
_DTYPE = torch.bfloat16


def _meta_inputs(shape):
    return [
        torch.randn(*shape, device="meta", dtype=_DTYPE),
        torch.randn(*shape, device="meta", dtype=_DTYPE),
        torch.randn(shape[-1], device="meta", dtype=_DTYPE),
    ]


def _register(pattern_cls, eps, pm_pass, shape):
    """Register one pattern instance with CPU-safe (meta) example inputs.

    Example inputs only feed the registration-time trace (metadata-only), so
    meta tensors keep this UT free of NPU allocations.
    """
    pattern = pattern_cls(SimpleNamespace(model_config=SimpleNamespace(dtype=_DTYPE)), eps=eps)
    pattern.get_inputs = lambda: _meta_inputs(shape)
    pattern.register(pm_pass)


def _build_graph(fn, shapes):
    """Fake-trace ``fn`` on CPU into the pass-visible (post-AOT) FX form."""
    mode = FakeTensorMode()
    with mode:
        args = tuple(torch.empty(s, dtype=_DTYPE) for s in shapes)
        gm = proxy_tensor.make_fx(fn, tracing_mode="fake")(*args)
    gm.graph.eliminate_dead_code()
    gm.recompile()
    return gm


def _direct_chain(eps):
    def fn(x, residual, weight):
        out = _NORM_OP(x, residual, weight, None, eps)
        quantized = torch.ops.npu.npu_dynamic_quant(out[0], dst_type=torch.int8)
        return quantized[0], quantized[1], out[2]

    return fn


def _view_chain(eps, sizes, view_op):
    def fn(x, residual, weight):
        out = _NORM_OP(x, residual, weight, None, eps)
        viewed = view_op(out[0], sizes)
        quantized = torch.ops.npu.npu_dynamic_quant(viewed, dst_type=torch.int8)
        return quantized[0], quantized[1], out[2]

    return fn


def _first_output_shape(gm):
    out_node = next(n for n in gm.graph.nodes if n.op == "output")
    first_arg = out_node.args[0][0]
    return tuple(first_arg.meta["tensor_meta"].shape)


_2D = ((2, 8), (2, 8), (8,))
_3D = ((2, 4, 8), (2, 4, 8), (8,))


class TestAddRMSNormDynamicQuantWithNoneBias(TestBase):
    def test_direct_chain_default_eps_fuses(self):
        """The 0.6B form: bias=None + default eps=1e-6, direct norm->quant."""
        from vllm_ascend.compilation.passes.norm_quant_fusion_pass import (
            AddRMSNormDynamicQuantPatternWithNoneBias,
            AddRMSNormDynamicQuantPatternWithNoneBiasView,
        )

        pm_pass = pm.PatternMatcherPass(pass_name="w2_direct")
        _register(AddRMSNormDynamicQuantPatternWithNoneBias, 1e-6, pm_pass, (2, 8))
        _register(AddRMSNormDynamicQuantPatternWithNoneBiasView, 1e-6, pm_pass, (2, 8))

        gm = _build_graph(_direct_chain(1e-6), _2D)
        count = pm_pass.apply(gm)

        self.assertEqual(count, 1)
        fused = [n for n in gm.graph.nodes if n.target is _FUSED_OP]
        self.assertEqual(len(fused), 1)
        # residual (output[2]) must still be exposed for downstream consumers
        self.assertEqual(len(gm.graph.find_nodes(op="output")), 1)

    def test_direct_chain_epsilon_discrimination(self):
        """Non-default eps=1e-5 keeps the full positional args (..., None, 1e-5);
        the 1e-6 pattern must not match it, the 1e-5 instances must."""
        from vllm_ascend.compilation.passes.norm_quant_fusion_pass import (
            AddRMSNormDynamicQuantPatternWithNoneBias,
            AddRMSNormDynamicQuantPatternWithNoneBiasView,
        )

        wrong_eps = pm.PatternMatcherPass(pass_name="w2_wrong_eps")
        _register(AddRMSNormDynamicQuantPatternWithNoneBias, 1e-6, wrong_eps, (2, 8))
        gm_wrong = _build_graph(_direct_chain(1e-5), _2D)
        self.assertEqual(wrong_eps.apply(gm_wrong), 0)

        right_eps = pm.PatternMatcherPass(pass_name="w2_right_eps")
        _register(AddRMSNormDynamicQuantPatternWithNoneBias, 1e-5, right_eps, (2, 8))
        _register(AddRMSNormDynamicQuantPatternWithNoneBiasView, 1e-5, right_eps, (2, 8))
        gm_right = _build_graph(_direct_chain(1e-5), _2D)
        self.assertEqual(right_eps.apply(gm_right), 1)

    def test_view_chain_fuses_and_restores_shape(self):
        """3D norm output -> view(-1, last) -> quant: view swallowed, merged
        shape restored by the replacement (flatten(0, -2) re-traced as a view
        with the concrete merged size)."""
        from vllm_ascend.compilation.passes.norm_quant_fusion_pass import (
            AddRMSNormDynamicQuantPatternWithNoneBias,
            AddRMSNormDynamicQuantPatternWithNoneBiasView,
        )

        pm_pass = pm.PatternMatcherPass(pass_name="w2_view")
        _register(AddRMSNormDynamicQuantPatternWithNoneBias, 1e-6, pm_pass, (2, 4, 8))
        _register(AddRMSNormDynamicQuantPatternWithNoneBiasView, 1e-6, pm_pass, (2, 4, 8))

        gm = _build_graph(_view_chain(1e-6, [-1, 8], torch.ops.aten.view), _3D)
        count = pm_pass.apply(gm)

        self.assertEqual(count, 1)
        self.assertEqual(len([n for n in gm.graph.nodes if n.target is _FUSED_OP]), 1)
        self.assertEqual(_first_output_shape(gm), (8, 8))

    def test_reshape_chain_fuses(self):
        """aten.reshape spelling (post-grad graphs record dynamic-size .view()
        calls as reshape) hits the reshape tree of the same pattern."""
        from vllm_ascend.compilation.passes.norm_quant_fusion_pass import (
            AddRMSNormDynamicQuantPatternWithNoneBias,
            AddRMSNormDynamicQuantPatternWithNoneBiasView,
        )

        pm_pass = pm.PatternMatcherPass(pass_name="w2_reshape")
        _register(AddRMSNormDynamicQuantPatternWithNoneBias, 1e-6, pm_pass, (2, 4, 8))
        _register(AddRMSNormDynamicQuantPatternWithNoneBiasView, 1e-6, pm_pass, (2, 4, 8))

        gm = _build_graph(_view_chain(1e-6, [-1, 8], torch.ops.aten.reshape), _3D)
        self.assertEqual(pm_pass.apply(gm), 1)
        self.assertEqual(len([n for n in gm.graph.nodes if n.target is _FUSED_OP]), 1)

    def test_view_shape_check_rejects_non_merge_views(self):
        """The extra_check must fail closed on views that are not a pure
        leading-dims merge: last-dim change and rank-preserving 3D view."""
        from vllm_ascend.compilation.passes.norm_quant_fusion_pass import (
            AddRMSNormDynamicQuantPatternWithNoneBias,
            AddRMSNormDynamicQuantPatternWithNoneBiasView,
        )

        pm_pass = pm.PatternMatcherPass(pass_name="w2_negative")
        _register(AddRMSNormDynamicQuantPatternWithNoneBias, 1e-6, pm_pass, (2, 4, 8))
        _register(AddRMSNormDynamicQuantPatternWithNoneBiasView, 1e-6, pm_pass, (2, 4, 8))

        last_dim_changed = _build_graph(_view_chain(1e-6, [-1, 4], torch.ops.aten.view), _3D)
        self.assertEqual(pm_pass.apply(last_dim_changed), 0)

        rank_preserving = _build_graph(_view_chain(1e-6, [2, 4, 8], torch.ops.aten.view), _3D)
        self.assertEqual(pm_pass.apply(rank_preserving), 0)

    def test_pass_registers_and_fuses_dynamic_chain(self):
        """Full AddRMSNormQuantFusionPass wiring: with enable_custom_op() the
        pass registers the new variants and fuses the 0.6B dynamic form."""

        def _ensure_vllm_quantize_op():
            # torch.ops.vllm.quantize is registered by vllm's native libs at
            # engine startup; the pre-existing static patterns only need the
            # schema at registration-time trace here, so define a stub when
            # absent (it is never executed by this UT).
            if hasattr(torch.ops.vllm, "quantize"):
                return
            lib = torch.library.Library("vllm", "FRAGMENT")
            lib.define("quantize(Tensor input, Tensor scale, Tensor scale_reciprocal, Tensor offset) -> Tensor")

            @torch.library.register_fake("vllm::quantize")
            def _(_input, _scale, _scale_reciprocal, _offset):
                return torch.empty_like(_input, dtype=torch.int8)

        from vllm.config import VllmConfig

        from vllm_ascend.compilation.passes.norm_quant_fusion_pass import AddRMSNormQuantFusionPass

        _ensure_vllm_quantize_op()
        real_randn = torch.randn

        def _meta_randn(*args, **kwargs):
            kwargs["device"] = "meta"
            return real_randn(*args, **kwargs)

        with (
            patch("torch.randn", _meta_randn),
            patch("vllm_ascend.platform.NPUPlatform.check_and_update_config"),
        ):
            vllm_config = VllmConfig()
            vllm_config.model_config = SimpleNamespace(dtype=_DTYPE)
            fusion_pass = AddRMSNormQuantFusionPass(vllm_config, dynamic_quant_fusion=True)

        self.assertEqual(fusion_pass._uuid_factors.get("custom_op"), "True")
        self.assertEqual(fusion_pass._uuid_factors.get("dyn_fusion"), "True")

        gm = _build_graph(_direct_chain(1e-6), _2D)
        fusion_pass(gm.graph)

        self.assertGreaterEqual(fusion_pass.matched_count, 1)
        self.assertEqual(len([n for n in gm.graph.nodes if n.target is _FUSED_OP]), 1)

    def test_pass_default_off_keeps_dynamic_chain_unfused(self):
        """方案 A (U-approved): fuse_norm_quant_dynamic defaults to False, so
        the pass leaves the dynamic chain untouched — both tracks' outputs stay
        bit-identical to today (test_w8a8_inductor_track_matches_eager parity).
        """

        def _ensure_vllm_quantize_op():
            if hasattr(torch.ops.vllm, "quantize"):
                return
            lib = torch.library.Library("vllm", "FRAGMENT")
            lib.define("quantize(Tensor input, Tensor scale, Tensor scale_reciprocal, Tensor offset) -> Tensor")

            @torch.library.register_fake("vllm::quantize")
            def _(_input, _scale, _scale_reciprocal, _offset):
                return torch.empty_like(_input, dtype=torch.int8)

        from vllm.config import VllmConfig

        from vllm_ascend.compilation.passes.norm_quant_fusion_pass import AddRMSNormQuantFusionPass

        _ensure_vllm_quantize_op()
        real_randn = torch.randn

        def _meta_randn(*args, **kwargs):
            kwargs["device"] = "meta"
            return real_randn(*args, **kwargs)

        with (
            patch("torch.randn", _meta_randn),
            patch("vllm_ascend.platform.NPUPlatform.check_and_update_config"),
        ):
            vllm_config = VllmConfig()
            vllm_config.model_config = SimpleNamespace(dtype=_DTYPE)
            fusion_pass = AddRMSNormQuantFusionPass(vllm_config)

        self.assertEqual(fusion_pass._uuid_factors.get("dyn_fusion"), "False")

        gm = _build_graph(_direct_chain(1e-6), _2D)
        fusion_pass(gm.graph)

        self.assertEqual(fusion_pass.matched_count, 0)
        self.assertEqual(len([n for n in gm.graph.nodes if n.target is _FUSED_OP]), 0)
