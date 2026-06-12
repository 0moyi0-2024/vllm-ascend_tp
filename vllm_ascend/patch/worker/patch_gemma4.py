import torch
from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
from vllm_ascend.ascend_forward_context import _EXTRA_CTX


def _get_local_mask(hidden_states):
    mc2_mask = getattr(_EXTRA_CTX, "mc2_mask", None)
    if mc2_mask is None:
        return None
    tp_size = get_tensor_model_parallel_world_size()
    if tp_size == 1:
        if mc2_mask.numel() == hidden_states.shape[0]:
            return mc2_mask.to(device=hidden_states.device, dtype=torch.bool)
        return None
    local_len = hidden_states.shape[0]
    if mc2_mask.numel() < local_len * tp_size:
        return None
    tp_rank = get_tensor_model_parallel_rank()
    local_mc2_mask = mc2_mask[tp_rank * local_len:(tp_rank + 1) * local_len]
    return local_mc2_mask.to(device=hidden_states.device, dtype=torch.bool)


def _patched_gemma4_decoder_layer_forward(original_forward):
    def new_forward(self, positions, hidden_states, residual=None, per_layer_input=None, **kwargs):
        mask = _get_local_mask(hidden_states)

        if mask is not None:
            hidden_states = torch.where(mask[:, None], hidden_states, torch.zeros_like(hidden_states))

        result = original_forward(self, positions, hidden_states, residual, per_layer_input=per_layer_input, **kwargs)

        if mask is not None and mask.numel() == result.shape[0]:
            result = torch.where(mask[:, None], result, torch.zeros_like(result))

        return result

    return new_forward


def apply_patch():
    from vllm.model_executor.models.gemma4 import Gemma4DecoderLayer

    original_forward = Gemma4DecoderLayer.forward
    Gemma4DecoderLayer.forward = _patched_gemma4_decoder_layer_forward(original_forward)


apply_patch()
