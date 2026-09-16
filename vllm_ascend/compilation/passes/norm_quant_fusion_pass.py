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
import functools
import operator

import torch
from torch._inductor.pattern_matcher import (
    MULTIPLE,
    CallFunction,
    Ignored,
    KeywordArg,
    Match,
    MultiOutputPattern,
    PatternMatcherPass,
    fwd_only,
    register_replacement,
)
from vllm.compilation.passes.inductor_pass import InductorPass
from vllm.compilation.passes.vllm_inductor_pass import (
    VllmInductorPass,
    VllmPatternMatcherPass,
)
from vllm.config import VllmConfig
from vllm.config.compilation import Range
from vllm.logger import logger

from vllm_ascend.compilation.passes.base_pattern import BasePattern
from vllm_ascend.utils import AscendDeviceType, enable_custom_op, get_ascend_device_type


class AddRMSNormQuantPattern(BasePattern):
    def __init__(self, vllm_config: VllmConfig, eps: float = 1e-6):
        super().__init__(vllm_config, eps)

    def get_inputs(self):
        """
        Generate example inputs for the AddRMSNormQuant fusion pattern.
        """
        rms_norm_input = torch.randn(2, 4, device="npu", dtype=self.dtype)
        residual = torch.randn(2, 4, device="npu", dtype=self.dtype)
        rms_norm_weight = torch.randn(4, device="npu", dtype=self.dtype)
        scale = torch.ones(4, device="npu", dtype=self.dtype)
        scale_reciprocal = torch.ones(4, device="npu", dtype=self.dtype)
        offset = torch.zeros(4, device="npu", dtype=self.dtype)
        return [rms_norm_input, residual, rms_norm_weight, scale, scale_reciprocal, offset]

    def get_pattern(self):
        def pattern(
            rms_norm_input: torch.Tensor,
            residual: torch.Tensor,
            rms_norm_weight: torch.Tensor,
            scale: torch.Tensor,
            scale_reciprocal: torch.Tensor,
            offset: torch.Tensor,
        ):
            """
            Pattern for AddRMSNormQuant fusion.
            """
            output = torch.ops._C_ascend.npu_add_rms_norm_bias(
                rms_norm_input, residual, rms_norm_weight, None, self.eps
            )
            out0 = output[0]
            out1 = output[2]
            quantized_output = torch.ops.vllm.quantize(out0, scale, scale_reciprocal, offset)
            return quantized_output, out1

        return pattern

    def get_replacement(self):
        def replacement(
            rms_norm_input: torch.Tensor,
            residual: torch.Tensor,
            rms_norm_weight: torch.Tensor,
            scale: torch.Tensor,
            scale_reciprocal: torch.Tensor,
            offset: torch.Tensor,
        ):
            """
            Replacement for the AddRMSNormQuant fusion.
            """
            output = torch.ops.npu.npu_add_rms_norm_quant(
                rms_norm_input, residual, rms_norm_weight, scale, offset, epsilon=self.eps
            )
            quantized_output = output[0]
            out1 = output[2]
            return quantized_output, out1

        return replacement


class AddRMSNormQuantPatternWithBias(BasePattern):
    def __init__(self, vllm_config: VllmConfig, eps: float = 1e-6):
        super().__init__(vllm_config, eps)

    def get_inputs(self):
        """
        Generate example inputs for the AddRMSNormQuant fusion pattern.
        """
        rms_norm_input = torch.randn(2, 4, device="npu", dtype=self.dtype)
        residual = torch.randn(2, 4, device="npu", dtype=self.dtype)
        rms_norm_weight = torch.randn(4, device="npu", dtype=self.dtype)
        rmsnorm_bias = torch.randn(4, device="npu", dtype=self.dtype)
        scale = torch.ones(4, device="npu", dtype=self.dtype)
        scale_reciprocal = torch.ones(4, device="npu", dtype=self.dtype)
        offset = torch.zeros(4, device="npu", dtype=self.dtype)
        return [rms_norm_input, residual, rms_norm_weight, scale, scale_reciprocal, offset, rmsnorm_bias]

    def get_pattern(self):
        def pattern(
            rms_norm_input: torch.Tensor,
            residual: torch.Tensor,
            rms_norm_weight: torch.Tensor,
            scale: torch.Tensor,
            scale_reciprocal: torch.Tensor,
            offset: torch.Tensor,
            bias: torch.Tensor,
        ):
            """
            Pattern for AddRMSNormQuant fusion.
            """
            output = torch.ops._C_ascend.npu_add_rms_norm_bias(
                rms_norm_input, residual, rms_norm_weight, bias, self.eps
            )
            out0 = output[0]
            out1 = output[2]
            quantized_output = torch.ops.vllm.quantize(out0, scale, scale_reciprocal, offset)
            return quantized_output, out1

        return pattern

    def get_replacement(self):
        def replacement(
            rms_norm_input: torch.Tensor,
            residual: torch.Tensor,
            rms_norm_weight: torch.Tensor,
            scale: torch.Tensor,
            scale_reciprocal: torch.Tensor,
            offset: torch.Tensor,
            bias: torch.Tensor,
        ):
            """
            Replacement for the AddRMSNormQuant fusion.
            """
            output = torch.ops.npu.npu_add_rms_norm_quant(
                rms_norm_input, residual, rms_norm_weight, scale, offset, epsilon=self.eps, beta=bias
            )
            quantized_output = output[0]
            out1 = output[2]
            return quantized_output, out1

        return replacement


class AddRMSNormDynamicQuantPattern(BasePattern):
    def __init__(self, vllm_config: VllmConfig, eps: float = 1e-6):
        super().__init__(vllm_config, eps)

    def get_inputs(self):
        """
        Generate example inputs for the AddRMSNormQuant fusion pattern.
        """
        rms_norm_input = torch.randn(2, 4, device="npu", dtype=self.dtype)
        residual = torch.randn(2, 4, device="npu", dtype=self.dtype)
        rms_norm_weight = torch.randn(4, device="npu", dtype=self.dtype)
        return [rms_norm_input, residual, rms_norm_weight]

    def get_pattern(self):
        def pattern(rms_norm_input: torch.Tensor, residual: torch.Tensor, rms_norm_weight: torch.Tensor):
            """
            Pattern for AddRMSNormQuant fusion.
            """
            output = torch.ops.npu.npu_add_rms_norm(rms_norm_input, residual, rms_norm_weight, self.eps)
            out0 = output[0]
            out1 = output[2]
            quantized_output = torch.ops.npu.npu_dynamic_quant(out0)
            return quantized_output[0], quantized_output[1], out1

        return pattern

    def get_replacement(self):
        def replacement(rms_norm_input: torch.Tensor, residual: torch.Tensor, rms_norm_weight: torch.Tensor):
            """
            Replacement for the AddRMSNormQuant fusion.
            """
            output = torch.ops.npu.npu_add_rms_norm_dynamic_quant(
                rms_norm_input, residual, rms_norm_weight, epsilon=self.eps, output_mask=[True, False]
            )
            return (
                output[0],
                output[3],
                output[2],
            )

        return replacement


class AddRMSNormDynamicQuantPatternWithBias(BasePattern):
    def __init__(self, vllm_config: VllmConfig, eps: float = 1e-6):
        super().__init__(vllm_config, eps)

    def get_inputs(self):
        """
        Generate example inputs for the AddRMSNormQuant fusion pattern.
        """
        rms_norm_input = torch.randn(2, 4, device="npu", dtype=self.dtype)
        residual = torch.randn(2, 4, device="npu", dtype=self.dtype)
        rms_norm_weight = torch.randn(4, device="npu", dtype=self.dtype)
        rmsnorm_bias = torch.randn(4, device="npu", dtype=self.dtype)
        return [rms_norm_input, residual, rms_norm_weight, rmsnorm_bias]

    def get_pattern(self):
        def pattern(
            rms_norm_input: torch.Tensor,
            residual: torch.Tensor,
            rms_norm_weight: torch.Tensor,
            bias: torch.Tensor,
        ):
            """
            Pattern for AddRMSNormQuant fusion.
            """
            output = torch.ops._C_ascend.npu_add_rms_norm_bias(
                rms_norm_input, residual, rms_norm_weight, bias, self.eps
            )
            out0 = output[0]
            out1 = output[2]
            quantized_output = torch.ops.npu.npu_dynamic_quant(out0)
            return quantized_output[0], quantized_output[1], out1

        return pattern

    def get_replacement(self):
        def replacement(
            rms_norm_input: torch.Tensor,
            residual: torch.Tensor,
            rms_norm_weight: torch.Tensor,
            bias: torch.Tensor,
        ):
            """
            Replacement for the AddRMSNormQuant fusion.
            """
            output = torch.ops.npu.npu_add_rms_norm_dynamic_quant(
                rms_norm_input, residual, rms_norm_weight, epsilon=self.eps, output_mask=[True, False], beta=bias
            )
            return (
                output[0],
                output[3],
                output[2],
            )

        return replacement


class AddRMSNormDynamicQuantPatternWithNoneBias(BasePattern):
    """Dynamic-quant variant for the ``bias=None`` custom-op norm form (stage4 W2, D1①).

    With ``enable_custom_op()`` the Add+RMSNorm node is
    ``_C_ascend.npu_add_rms_norm_bias(x, residual, weight, None, eps)`` — the 4th
    arg is the constant ``None`` (0.6B W8A8_DYNAMIC graph, T0b-1R). The existing
    ``AddRMSNormDynamicQuantPatternWithBias`` requires a tensor there and cannot
    match a constant, so this variant bakes ``None`` like the static
    ``AddRMSNormQuantPattern`` does.
    """

    def __init__(self, vllm_config: VllmConfig, eps: float = 1e-6):
        super().__init__(vllm_config, eps)

    def get_inputs(self):
        """
        Generate example inputs for the AddRMSNormDynamicQuant fusion pattern.
        """
        rms_norm_input = torch.randn(2, 4, device="npu", dtype=self.dtype)
        residual = torch.randn(2, 4, device="npu", dtype=self.dtype)
        rms_norm_weight = torch.randn(4, device="npu", dtype=self.dtype)
        return [rms_norm_input, residual, rms_norm_weight]

    def get_pattern(self):
        def pattern(rms_norm_input: torch.Tensor, residual: torch.Tensor, rms_norm_weight: torch.Tensor):
            """
            Pattern for AddRMSNormDynamicQuant fusion (bias=None custom-op form).
            """
            output = torch.ops._C_ascend.npu_add_rms_norm_bias(
                rms_norm_input, residual, rms_norm_weight, None, self.eps
            )
            out0 = output[0]
            out1 = output[2]
            quantized_output = torch.ops.npu.npu_dynamic_quant(out0)
            return quantized_output[0], quantized_output[1], out1

        return pattern

    def get_replacement(self):
        def replacement(rms_norm_input: torch.Tensor, residual: torch.Tensor, rms_norm_weight: torch.Tensor):
            """
            Replacement for AddRMSNormDynamicQuant fusion (bias=None custom-op form).
            """
            output = torch.ops.npu.npu_add_rms_norm_dynamic_quant(
                rms_norm_input, residual, rms_norm_weight, epsilon=self.eps, output_mask=[True, False]
            )
            return (
                output[0],
                output[3],
                output[2],
            )

        return replacement


def _merged_leading_dims_view_check(match: Match) -> bool:
    """Extra check for the view/reshape-swallowing pattern (stage4 W2, D1②).

    The replacement restores the swallowed shape via ``flatten(0, -2)``, so the
    matched view must be a pure leading-dims merge: 2D output with the last dim
    unchanged and the leading element count preserved. Fails closed when the
    shape facts are missing or a symbolic comparison cannot be decided locally.
    """
    view_node = None
    for node in match.nodes:
        if node.target in (torch.ops.aten.view.default, torch.ops.aten.reshape.default):
            view_node = node
            break
    if view_node is None or not isinstance(view_node.args[0], torch.fx.Node):
        return False
    norm_meta = view_node.args[0].meta.get("tensor_meta")
    view_meta = view_node.meta.get("tensor_meta")
    if norm_meta is None or view_meta is None:
        return False
    norm_shape, view_shape = norm_meta.shape, view_meta.shape
    try:
        if len(view_shape) != 2 or len(norm_shape) < 2:
            return False
        if view_shape[-1] != norm_shape[-1]:
            return False
        return functools.reduce(operator.mul, view_shape[:-1], 1) == functools.reduce(operator.mul, norm_shape[:-1], 1)
    except Exception:  # symbolic shape comparison not locally decidable -> fail closed
        return False


class AddRMSNormDynamicQuantPatternWithNoneBiasView(BasePattern):
    """View/reshape-swallowing ``bias=None`` dynamic-quant variant (stage4 W2, D1②).

    Matches the same ``_C_ascend.npu_add_rms_norm_bias(..., None[, eps])`` node
    when the norm output feeds an ``aten.view``/``aten.reshape`` (leading-dims
    merge) before ``npu_dynamic_quant`` — the 3D-input W8A8_DYNAMIC form (the
    view only shows up inside wrapper layers, T0b-1R; TN npugraph registers the
    same chain for npu_add_rms_norm).

    A traced pattern cannot express this: tracing bakes the concrete view size
    (``[-1, 4]`` from the example inputs) and the specific-match phase compares
    sizes exactly, so real hidden sizes would never match. The search pattern is
    therefore hand-written as a ``CallFunction`` tree with ``Ignored()`` view
    size (TN npugraph ``add_rms_norm_dynamic_quant.py`` precedent) plus the
    shape-preserving ``_merged_leading_dims_view_check``.
    """

    def __init__(self, vllm_config: VllmConfig, eps: float = 1e-6):
        super().__init__(vllm_config, eps)

    def get_inputs(self):
        """
        Generate example inputs for the AddRMSNormDynamicQuant fusion pattern.
        """
        rms_norm_input = torch.randn(2, 4, device="npu", dtype=self.dtype)
        residual = torch.randn(2, 4, device="npu", dtype=self.dtype)
        rms_norm_weight = torch.randn(4, device="npu", dtype=self.dtype)
        return [rms_norm_input, residual, rms_norm_weight]

    def get_pattern(self):
        # Signature carrier only: register() below passes the hand-written
        # search_fn_pattern, so this body is never traced (TN npugraph uses the
        # same stub style for its hand-written patterns).
        def pattern(rms_norm_input: torch.Tensor, residual: torch.Tensor, rms_norm_weight: torch.Tensor):
            pass

        return pattern

    def get_replacement(self):
        def replacement(rms_norm_input: torch.Tensor, residual: torch.Tensor, rms_norm_weight: torch.Tensor):
            """
            Replacement for AddRMSNormDynamicQuant fusion with a swallowed view:
            the fused output is re-flattened to restore the viewed shape.
            """
            output = torch.ops.npu.npu_add_rms_norm_dynamic_quant(
                rms_norm_input, residual, rms_norm_weight, epsilon=self.eps, output_mask=[True, False]
            )
            return (
                output[0].flatten(0, -2),
                output[3],
                output[2],
            )

        return replacement

    def get_search_fn_patterns(self) -> list[MultiOutputPattern]:
        """Hand-written search patterns for both view spellings.

        fx tracing drops schema-default trailing args, so the default-eps node
        form is ``(x1, x2, gamma)`` while a non-default epsilon keeps the full
        positional form ``(x1, x2, gamma, None, eps)``; both traced shapes are
        mirrored here (verified against fake-tensor traces of the real chain).
        """
        norm_op = torch.ops._C_ascend.npu_add_rms_norm_bias.default
        eps_default = next(arg.default_value for arg in norm_op._schema.arguments if arg.name == "epsilon")
        norm_args: list = [
            KeywordArg("rms_norm_input"),
            KeywordArg("residual"),
            KeywordArg("rms_norm_weight"),
        ]
        if self.eps != eps_default:
            norm_args += [None, self.eps]
        patterns = []
        for view_op in (torch.ops.aten.view.default, torch.ops.aten.reshape.default):
            # _users=MULTIPLE on the norm node: fx materializes every tuple
            # element, so the (dead) rstd getitem may or may not have been DCE'd
            # by the time this pass runs; either user count must be accepted.
            norm_func = CallFunction(norm_op, *norm_args, _users=MULTIPLE)
            norm_out0 = CallFunction(operator.getitem, norm_func, 0)
            norm_out2 = CallFunction(operator.getitem, norm_func, 2)
            quant_func = CallFunction(
                torch.ops.npu.npu_dynamic_quant.default,
                CallFunction(view_op, norm_out0, Ignored()),
                _users=2,
            )
            patterns.append(
                MultiOutputPattern(
                    [
                        CallFunction(operator.getitem, quant_func, 0),
                        CallFunction(operator.getitem, quant_func, 1),
                        norm_out2,
                    ]
                )
            )
        return patterns

    def register(self, pm_pass: PatternMatcherPass) -> None:
        # BasePattern.register cannot express the two knobs this pattern needs
        # (hand-written search pattern + extra_check). The nge global-table leg
        # of BasePattern.register is skipped on purpose: that API has no
        # search_fn_pattern slot, and a traced fallback would pin concrete view
        # sizes (never matching real hidden sizes).
        for search_pattern in self.get_search_fn_patterns():
            register_replacement(
                self.get_pattern(),
                self.get_replacement(),
                self.get_inputs(),
                fwd_only,
                pm_pass,
                extra_check=_merged_leading_dims_view_check,
                search_fn_pattern=search_pattern,
            )


class AddRMSNormDynamicMXQuantPattern(BasePattern):
    def __init__(self, vllm_config: VllmConfig, eps: float = 1e-6):
        super().__init__(vllm_config, eps)

    def get_inputs(self):
        """
        Generate example inputs for the AddRMSNormDynamicMXQuant fusion pattern.
        """
        rms_norm_input = torch.randn(2, 64, device="npu", dtype=self.dtype)
        residual = torch.randn(2, 64, device="npu", dtype=self.dtype)
        rms_norm_weight = torch.randn(64, device="npu", dtype=self.dtype)
        return [rms_norm_input, residual, rms_norm_weight]

    def get_pattern(self):
        def pattern(rms_norm_input: torch.Tensor, residual: torch.Tensor, rms_norm_weight: torch.Tensor):
            """
            Pattern for AddRMSNormDynamicMXQuant fusion.
            """
            output = torch.ops.npu.npu_add_rms_norm(rms_norm_input, residual, rms_norm_weight, self.eps)
            out0 = output[0]
            out1 = output[2]
            quantized_output = torch.ops.npu.npu_dynamic_mx_quant(out0, dst_type=torch.float8_e4m3fn)
            return quantized_output[0], quantized_output[1], out1

        return pattern

    def get_replacement(self):
        def replacement(rms_norm_input: torch.Tensor, residual: torch.Tensor, rms_norm_weight: torch.Tensor):
            """
            Replacement for the AddRMSNormDynamicMXQuant fusion.
            """
            output = torch.ops.npu.npu_add_rms_norm_dynamic_mx_quant(
                rms_norm_input,
                residual,
                rms_norm_weight,
                epsilon=self.eps,
                dst_type=torch.float8_e4m3fn,
            )
            return (
                output[0],
                output[2],
                output[1],
            )

        return replacement


class RMSNormDynamicMXQuantPattern(BasePattern):
    def __init__(self, vllm_config: VllmConfig, eps: float = 1e-6):
        super().__init__(vllm_config, eps)

    def get_inputs(self):
        """
        Generate example inputs for the RMSNormDynamicMXQuant fusion pattern.
        """
        rms_norm_input = torch.randn(2, 64, device="npu", dtype=self.dtype)
        rms_norm_weight = torch.randn(64, device="npu", dtype=self.dtype)
        return [rms_norm_input, rms_norm_weight]

    def get_pattern(self):
        def pattern(rms_norm_input: torch.Tensor, rms_norm_weight: torch.Tensor):
            """
            Pattern for RMSNormDynamicMXQuant fusion.
            """
            output = torch.ops.npu.npu_rms_norm(rms_norm_input, rms_norm_weight, self.eps)
            out0 = output[0]
            quantized_output = torch.ops.npu.npu_dynamic_mx_quant(out0, dst_type=torch.float8_e4m3fn)
            return quantized_output[0], quantized_output[1]

        return pattern

    def get_replacement(self):
        def replacement(rms_norm_input: torch.Tensor, rms_norm_weight: torch.Tensor):
            """
            Replacement for the RMSNormDynamicMXQuant fusion.
            """
            output = torch.ops.npu.npu_rms_norm_dynamic_mx_quant(
                rms_norm_input,
                rms_norm_weight,
                epsilon=self.eps,
                dst_type=torch.float8_e4m3fn,
            )
            return output[0], output[1]

        return replacement


def _model_uses_w4a4_quant(vllm_config: VllmConfig | None) -> bool:
    """Check whether the model uses W4A4 int4 quantization for any layer.

    W4A4 int4 schemes (e.g. W4A4_DYNAMIC, W4A4_FLATQUANT_DYNAMIC)
    are incompatible with the fuse_norm_quant optimization,
    so callers use this to disable that fusion.
    """
    if vllm_config is None:
        return False
    quant_config = getattr(vllm_config, "quant_config", None)
    if quant_config is None:
        return False
    quant_description = getattr(quant_config, "quant_description", None)
    if not quant_description:
        return False
    w4a4_int4_schemes = ["W4A4_DYNAMIC", "W4A4_FLATQUANT_DYNAMIC"]
    return any(
        isinstance(quant_type, str) and quant_type in w4a4_int4_schemes for quant_type in quant_description.values()
    )


class AddRMSNormQuantFusionPass(VllmInductorPass):
    """
    A pass for fusing AddRMSNorm and W8A8 quantization operations on Ascend.
    """

    def __init__(self, vllm_config: VllmConfig, dynamic_quant_fusion: bool = False):
        super().__init__(vllm_config)
        self.pattern_match_passes: PatternMatcherPass = PatternMatcherPass(pass_name="rmsnorm_quant_fusion_pass")
        # uuid factors (stage3 03 R3-5): pattern-set branches must fold into the
        # cache key (dtype / W4A4 skip / A5 MX branch / enable_custom_op).
        self._uuid_factors: dict = {
            "dtype": str(vllm_config.model_config.dtype),
            "dyn_fusion": str(dynamic_quant_fusion),
        }

        dtype = vllm_config.model_config.dtype
        if dtype not in (torch.bfloat16, torch.float16):
            logger.debug("Quant fusion not enabled: unsupported dtype %s", dtype)
            return

        if _model_uses_w4a4_quant(vllm_config):
            logger.debug(
                "Quant fusion not enabled: the model contains "
                "W4A4 quantized weights, which are incompatible with the "
                "norm-quant fusion pass."
            )
            self._uuid_factors["w4a4_skip"] = "1"
            return
        self._uuid_factors["w4a4_skip"] = "0"
        self._uuid_factors["device"] = str(get_ascend_device_type())
        self._uuid_factors["custom_op"] = str(enable_custom_op())

        common_epsilons = [1e-5, 1e-6]

        for eps in common_epsilons:
            AddRMSNormDynamicQuantPattern(vllm_config, eps=eps).register(self.pattern_match_passes)
            if get_ascend_device_type() == AscendDeviceType.A5:
                AddRMSNormDynamicMXQuantPattern(vllm_config, eps=eps).register(self.pattern_match_passes)
                RMSNormDynamicMXQuantPattern(vllm_config, eps=eps).register(self.pattern_match_passes)
            if enable_custom_op():
                AddRMSNormQuantPattern(vllm_config, eps=eps).register(self.pattern_match_passes)
                AddRMSNormQuantPatternWithBias(vllm_config, eps=eps).register(self.pattern_match_passes)
                AddRMSNormDynamicQuantPatternWithBias(vllm_config, eps=eps).register(self.pattern_match_passes)
                # stage4 W2 (D1): with enable_custom_op() the dynamic-quant graph
                # norm node is _C_ascend.npu_add_rms_norm_bias(..., None, eps)
                # (T0b-1R, 0.6B W8A8_DYNAMIC) — register the bias=None dynamic
                # variant, with and without a view/reshape between the norm
                # output and npu_dynamic_quant. Opt-in via
                # fuse_norm_quant_dynamic (方案 A, U-approved): the fused kernel
                # is not token-identical to the eager chain, so default-off
                # keeps both tracks' outputs bit-identical.
                if dynamic_quant_fusion:
                    AddRMSNormDynamicQuantPatternWithNoneBias(vllm_config, eps=eps).register(self.pattern_match_passes)
                    AddRMSNormDynamicQuantPatternWithNoneBiasView(vllm_config, eps=eps).register(self.pattern_match_passes)

    def uuid(self) -> str:
        return InductorPass.hash_dict({"src": super().uuid(), **self._uuid_factors})

    def __call__(self, graph: torch.fx.Graph):
        self.begin()
        self.matched_count = self.pattern_match_passes.apply(graph)
        VllmPatternMatcherPass.match_table[self.pattern_match_passes.pass_name] += self.matched_count
        logger.debug("Replaced %s patterns", self.matched_count)
        self.end_and_log()

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        """
        Check if the pass is applicable for the current configuration.
        """
        return True
