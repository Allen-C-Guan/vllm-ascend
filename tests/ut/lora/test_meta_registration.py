# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

# The schemas of ``torch.ops._C_ascend`` only exist once the compiled extension
# is loaded; CPU runners without it skip this module wholesale.
pytest.importorskip("vllm_ascend.vllm_ascend_C")
import vllm_ascend.meta_registration  # noqa: E402, F401  # registers Meta kernels


def _meta_registered(op_name: str) -> bool:
    registrations = torch._C._dispatch_get_registrations_for_dispatch_key("Meta")
    return f"_C_ascend::{op_name}" in registrations


def test_shrink_meta_registered() -> None:
    assert _meta_registered("bgmv_shrink")
    assert _meta_registered("sgmv_shrink")
    # The expand registrations share the same capability gate; guard them too.
    assert _meta_registered("bgmv_expand")
    assert _meta_registered("sgmv_expand")


def test_bgmv_shrink_callable_under_fake_tensor_mode() -> None:
    with FakeTensorMode():
        x = torch.empty(4, 16, dtype=torch.float16)
        weight = torch.empty(2, 8, 16, dtype=torch.float16)
        indices = torch.empty(4, dtype=torch.int64)
        y = torch.empty(4, 8, dtype=torch.float32)
        out = torch.ops._C_ascend.bgmv_shrink(x, weight, indices, y, 0.5)
    # Schema is ``-> ()``: the op mutates ``y`` in place and returns nothing.
    assert out is None
    assert y.shape == (4, 8)
    assert y.dtype == torch.float32


def test_sgmv_shrink_callable_under_fake_tensor_mode() -> None:
    with FakeTensorMode():
        x = torch.empty(4, 16, dtype=torch.float16)
        weight = torch.empty(2, 8, 16, dtype=torch.float16)
        lora_indices = torch.empty(2, dtype=torch.int64)
        seq_len = torch.empty(2, dtype=torch.int64)
        y = torch.empty(4, 8, dtype=torch.float32)
        out = torch.ops._C_ascend.sgmv_shrink(x, weight, lora_indices, seq_len, y, 0.5)
    assert out is None
    assert y.shape == (4, 8)
    assert y.dtype == torch.float32
