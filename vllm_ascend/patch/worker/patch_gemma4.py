import torch
from vllm_ascend.ascend_forward_context import _EXTRA_CTX


def _patched_gemma4_decoder_layer_forward(original_forward):
    def new_forward(self, positions, hidden_states, residual=None, per_layer_input=None, **kwargs):
        mc2_mask = getattr(_EXTRA_CTX, "mc2_mask", None)
        num_actual_tokens = getattr(_EXTRA_CTX, "num_actual_tokens", None)

        if mc2_mask is not None and mc2_mask.numel() == hidden_states.shape[0]:
            mask = mc2_mask.to(device=hidden_states.device, dtype=torch.bool)
            hidden_states = torch.where(mask[:, None], hidden_states, torch.zeros_like(hidden_states))

        result = original_forward(self, positions, hidden_states, residual, per_layer_input=per_layer_input, **kwargs)

        if mc2_mask is not None and mc2_mask.numel() == result.shape[0]:
            mask = mc2_mask.to(device=result.device, dtype=torch.bool)
            result = torch.where(mask[:, None], result, torch.zeros_like(result))

        return result

    return new_forward


def apply_patch():
    from vllm.model_executor.models.gemma4 import Gemma4DecoderLayer

    original_forward = Gemma4DecoderLayer.forward
    Gemma4DecoderLayer.forward = _patched_gemma4_decoder_layer_forward(original_forward)
