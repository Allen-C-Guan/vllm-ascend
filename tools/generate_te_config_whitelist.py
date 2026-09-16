#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
#
"""Generate the triton_experimental (TE) config key whitelist as JSON.

Stage-4 W3 / decision D4 (stage design/stage4/02_设计方案.md §三): the
whitelist is generated, never handwritten. The script imports
torch_npu._inductor first so the TE backend is activated (activation is also
what injects ``npu_backend`` as a legal torch inductor config key — its
presence in ``torch._inductor.config.get_config_copy()`` is patch-timing
sensitive), then exports the TE config key list for CI reconciliation and
documentation generation. Importing the config module needs no NPU device.

Usage:
    python tools/generate_te_config_whitelist.py             # JSON to stdout
    python tools/generate_te_config_whitelist.py --out f.json
"""

import argparse
import json
import sys

if __name__ == "__main__":
    # When run as a script, Python prepends this file's directory (tools/) to
    # sys.path; the tools/bisect/ subpackage then shadows the stdlib bisect
    # module imported transitively by torch (torch -> random -> bisect).
    # Nothing is imported from tools/ itself, so drop the entry.
    if sys.path and sys.path[0]:
        sys.path.pop(0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export the triton_experimental config key whitelist as JSON.")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Write JSON to this file instead of stdout.",
    )
    args = parser.parse_args()

    import torch
    import torch_npu
    import torch_npu._inductor  # noqa: F401  # activation: registers the TE backend
    from torch_npu._inductor.triton_experimental import config as te_config

    te_config_keys = sorted(te_config.get_config_copy())
    payload = {
        "generator": "tools/generate_te_config_whitelist.py",
        "torch_version": torch.__version__,
        "torch_npu_version": torch_npu.__version__,
        "te_config_keys": te_config_keys,
        "te_config_key_count": len(te_config_keys),
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
