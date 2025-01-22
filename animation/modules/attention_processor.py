import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.utils.import_utils import is_xformers_available
import pdb
import todos

if is_xformers_available():
    import xformers
else:
    print(1/0)

# xxxx_debug
class AnimationAttnProcessor(nn.Module):
    def __init__(
        self, 
        hidden_size=None,
        cross_attention_dim=None,
        rank=4,
        network_alpha=None,
        lora_scale=1.0,):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        # hidden_size = 320
        # cross_attention_dim = None
        # rank = 128
        # network_alpha = None
        # lora_scale = 1.0

    def __call__(self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        # attention_mask = None
        # temb = None
        # (Pdb) attn
        # Attention(
        #   (to_q): Linear(in_features=320, out_features=320, bias=False)
        #   (to_k): Linear(in_features=320, out_features=320, bias=False)
        #   (to_v): Linear(in_features=320, out_features=320, bias=False)
        #   (to_out): ModuleList(
        #     (0): Linear(in_features=320, out_features=320, bias=True)
        #     (1): Dropout(p=0.0, inplace=False)
        #   )
        #   (processor): AnimationAttnProcessor()
        # )
        # tensor [hidden_states] size: [16, 4096, 320], min: -0.878906, max: 1.689453, mean: 0.00032
        assert hidden_states is not None
        # assert encoder_hidden_states is not None
        # assert attention_mask == None
        assert temb == None

        residual = hidden_states
        input_ndim = hidden_states.ndim # 3

        # tensor [encoder_hidden_states] size: [16, 4096, 320], min: -2.890625, max: 3.320312, mean: 0.006327
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        # attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        # assert attention_mask == None

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            # ==> pdb.set_trace()
            encoder_hidden_states = hidden_states

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query).contiguous()
        key = attn.head_to_batch_dim(key).contiguous()
        value = attn.head_to_batch_dim(value).contiguous()

        # assert attention_mask == None
        if is_xformers_available(): # True
            ### xformers
            hidden_states = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=None)
            hidden_states = hidden_states.to(query.dtype)
        else:
            attention_probs = attn.get_attention_scores(query, key, None)
            hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        # hidden_states = attn.to_out[0](hidden_states) + self.lora_scale * self.to_out_lora(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

