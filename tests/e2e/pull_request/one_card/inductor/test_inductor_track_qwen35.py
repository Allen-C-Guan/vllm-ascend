import os
from unittest.mock import patch

import huggingface_hub
from huggingface_hub import snapshot_download as hf_snapshot_download
import pytest

from tests.e2e.conftest import wait_until_npu_memory_free
from tests.e2e.pull_request.utils import PROMPTS_SHORT, compare_logprobs


pytestmark = pytest.mark.e2e_model("Qwen/Qwen3.5-0.8B")


@wait_until_npu_memory_free(max_wait_seconds=900)
def test_qwen35_inductor_track_matches_eager():
    model_path = os.environ.get("QWEN35_08B_MODEL")
    if model_path is None:
        model_path = hf_snapshot_download(
            "Qwen/Qwen3.5-0.8B",
            local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
        )
    runner_kwargs = {
        "model_name": model_path,
        "dtype": "bfloat16",
        "max_model_len": 4096,
        "max_num_seqs": 16,
        "gpu_memory_utilization": 0.85,
        "compilation_config": {"backend": "inductor"},
    }
    with patch.dict(os.environ, {"VLLM_USE_BREAKABLE_CUDAGRAPH": "0"}):
        compare_logprobs(runner_kwargs=runner_kwargs, prompts=PROMPTS_SHORT)
