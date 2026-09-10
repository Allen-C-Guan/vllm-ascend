#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
"""PostGradPassManager for the inductor compile-backend track.

The upstream PostGradPassManager's always-on ``FixFunctionalizationPass``
builds its rope target list from CUDA-only ops (``torch.ops._C.rotary_embedding``,
``vllm/compilation/passes/utility/fix_functionalization.py``), which raises
AttributeError on NPU where the vLLM CUDA extension is absent. XPU carries an
early-return guard for the same reason; NPU does not. ``AscendPostGradPassManager``
keeps the upstream manager and swaps that one pass for a no-op (stage
design/stage1/03 R1, fallback plan F1).
"""

import torch
import torch.fx
from vllm.compilation.passes.pass_manager import PostGradPassManager
from vllm.compilation.passes.vllm_inductor_pass import VllmInductorPass
from vllm.logger import init_logger

logger = init_logger(__name__)


class AscendNoopInductorPass(VllmInductorPass):
    """A no-op pass used to replace passes that are unsafe on NPU.

    ``uuid()`` is inherited from ``InductorPass`` (source hash), so the
    inductor code cache keys change together with this implementation.
    """

    def __call__(self, graph: torch.fx.Graph) -> None:
        logger.debug("Skipping %s: replaced by AscendNoopInductorPass (not supported on NPU).", self.pass_name)


class AscendPostGradPassManager(PostGradPassManager):
    """Upstream PostGradPassManager with NPU-unsafe always-on passes disabled.

    Only ``fix_functionalization`` is replaced today; the remaining always-on
    chain (post_cleanup / ir_lowering / clone_elimination) is platform-neutral
    FX bookkeeping and is kept as-is.
    """

    def configure(self, config) -> None:
        super().configure(config)
        noop = AscendNoopInductorPass(config)
        noop.pass_name = "FixFunctionalizationPass(noop)"
        self.fix_functionalization = noop
