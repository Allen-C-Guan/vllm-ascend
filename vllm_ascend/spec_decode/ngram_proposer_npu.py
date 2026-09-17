import torch
from vllm.v1.spec_decode.ngram_proposer_gpu import NgramProposerGPU


class AscendNgramProposerNPU(NgramProposerGPU):
    def __init__(self, vllm_config, device: torch.device, runner):
        # Stage-4 #7/A2 (R17): do NOT call super().__init__(). The upstream
        # constructor builds a bare VllmConfig (no additional_config) whose
        # __post_init__ re-runs check_and_update_config and rebuilds the
        # AscendConfig singleton trackless, wiping the engine's compile-track
        # state for the main model's compiles (and its own _dummy_run compile
        # then reads the legacy pass_key, which is not a registered torch
        # config key on the inductor track -> AttributeError). This subclass
        # is a stub — propose/dummy_run/load_model are all no-ops and the
        # real ngram_gpu path is the runner-side triton_ngram_spec_decode
        # integration — so only the scalar attributes the runner reads need
        # to be set, sourced from the ENGINE config directly.
        spec = vllm_config.speculative_config
        self.vllm_config = vllm_config
        self.min_n = spec.prompt_lookup_min
        self.max_n = spec.prompt_lookup_max
        self.k = spec.num_speculative_tokens
        self.max_model_len = vllm_config.model_config.max_model_len
        self.max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        self.device = device

    def load_model(self, *args, **kwargs):
        # No model to load.
        pass

    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens,
        with_prefill=None,
        in_graph_capturing=None,
        num_reqs=None,
        num_tokens_across_dp=None,
        aclgraph_runtime_mode=None,
        batch_descriptor=None,
        dummy_compute_logits=lambda hidden_states: None,
        is_profile=False,
    ):
        pass

    def propose(
        self,
        num_tokens_no_spec: torch.Tensor,  # [batch_size]
        token_ids_gpu: torch.Tensor,  # [batch_size, max_len]
        valid_sampled_token_ids_gpu: torch.Tensor,  # [batch_size, num_spec_tokens + 1]
        valid_sampled_tokens_count: torch.Tensor,  # [batch_size]
    ):
        pass
