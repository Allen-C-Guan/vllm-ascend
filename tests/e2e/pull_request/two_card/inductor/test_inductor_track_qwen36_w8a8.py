import os
from unittest.mock import patch

import pytest

from tests.e2e.conftest import wait_until_npu_memory_free
from tests.e2e.pull_request.utils import PROMPTS_SHORT, compare_logprobs

_MODEL = os.environ.get(
    "QWEN36_35B_W8A8_MODEL", "/mnt/weight/Qwen3.6-35B-A3B-w8a8"
)

pytestmark = [
    pytest.mark.e2e_model("Eco-Tech/Qwen3.6-35B-A3B-w8a8"),
    pytest.mark.skipif(
        not os.path.isdir(_MODEL),
        reason=f"Qwen3.6-35B-A3B-W8A8 weights not found at {_MODEL}",
    ),
]


@wait_until_npu_memory_free(max_wait_seconds=900)
def test_qwen36_w8a8_inductor_track_matches_eager():
    runner_kwargs = {
        "model_name": _MODEL,
        "quantization": "ascend",
        "dtype": "bfloat16",
        "max_model_len": 4096,
        "max_num_seqs": 16,
        "tensor_parallel_size": 2,
        "enable_expert_parallel": False,
        "gpu_memory_utilization": 0.75,
        "distributed_executor_backend": "mp",
        "compilation_config": {"backend": "inductor"},
    }
    with patch.dict(os.environ, {"VLLM_USE_BREAKABLE_CUDAGRAPH": "0"}):
        compare_logprobs(runner_kwargs=runner_kwargs, prompts=PROMPTS_SHORT)
