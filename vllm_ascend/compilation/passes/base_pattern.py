from abc import ABC, abstractmethod
from collections.abc import Callable

import torch
import torch._inductor.pattern_matcher as pm
from torch._inductor.pattern_matcher import PatternMatcherPass
from vllm.config import VllmConfig
from vllm.logger import logger

try:
    import npugraph_ex as nge
except ImportError:
    try:
        import torchair as nge
    except ImportError:
        # stage3 R3-6: neither legacy graph package is installed — the torch
        # inductor pattern registration below still works; only the (unused on
        # the inductor track) nge global table is skipped. Degrade instead of
        # crashing configure.
        nge = None
        logger.warning(
            "npugraph_ex/torchair both unavailable: Ascend fusion patterns will "
            "register to torch inductor only (nge registration skipped)."
        )

from vllm_ascend.compilation.passes.utils.npugraph_ex_utils_check import extra_stream_scope_check

# Global set to track registered patterns and prevent duplicates
_registered_patterns: set[str] = set()


class BasePattern(ABC):
    def __init__(self, vllm_config: VllmConfig, eps: float = 1e-6):
        self.vllm_config = vllm_config
        self.dtype = vllm_config.model_config.dtype
        self.eps = eps

    @abstractmethod
    def get_inputs(self) -> list[torch.Tensor]:
        pass

    @abstractmethod
    def get_pattern(self) -> Callable:
        pass

    @abstractmethod
    def get_replacement(self) -> Callable:
        pass

    def get_extra_stream_scope_check(self):
        return extra_stream_scope_check

    def register(self, pm_pass: PatternMatcherPass) -> None:
        # Create a unique identifier for this pattern based on class name and eps
        pattern_id = f"{self.__class__.__name__}_{self.eps}"

        pattern_fn = self.get_pattern()
        replacement_fn = self.get_replacement()
        example_inputs = self.get_inputs()

        # torch's pattern dedup (`seen_patterns`) is per-PatternMatcherPass
        # instance, so every pass instance must register its own patterns —
        # skipping here would hand a second engine an empty pattern set
        # (stage3: multi-engine same-config silently lost all fusions).
        pm.register_replacement(pattern_fn, replacement_fn, example_inputs, pm.fwd_only, pm_pass)

        # The nge table is process-global: dedupe across instances (its pattern_id
        # lacks shape factors — see stage3 03 known-limitation R3-11).
        if nge is not None and pattern_id not in _registered_patterns:
            nge.register_replacement(
                search_fn=pattern_fn,
                replace_fn=replacement_fn,
                example_inputs=example_inputs,
                extra_check=self.get_extra_stream_scope_check(),
            )

        # Mark this pattern as registered
        _registered_patterns.add(pattern_id)
