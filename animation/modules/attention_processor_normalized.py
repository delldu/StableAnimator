import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.utils.import_utils import is_xformers_available
import pdb

if is_xformers_available():
    import xformers
else:
    print(1 / 0)

class AnimationIDAttnNormalizedProcessor(nn.Module):
    def __init__(
            self,
            hidden_size,
            cross_attention_dim=None,
            rank=4,
            network_alpha=None,
            lora_scale=1.0,
            scale=1.0,
            num_tokens=4):
        super().__init__()
        # hidden_size = 320
        # cross_attention_dim = 1024
        # rank = 128
        # network_alpha = None
        # lora_scale = 1.0
        # scale = 1.0
        # num_tokens = 4


        self.scale = scale
        self.id_to_k = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.id_to_v = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)

        self.num_tokens = num_tokens
        # self = AnimationIDAttnNormalizedProcessor(
        #   (id_to_k): Linear(in_features=1024, out_features=320, bias=False)
        #   (id_to_v): Linear(in_features=1024, out_features=320, bias=False)
        # )

    def __call__(
            self,
            attn,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            temb=None,
            scale=1.0,
    ):
        # (Pdb) attn
        # Attention(
        #   (to_q): Linear(in_features=320, out_features=320, bias=False)
        #   (to_k): Linear(in_features=1024, out_features=320, bias=False)
        #   (to_v): Linear(in_features=1024, out_features=320, bias=False)
        #   (to_out): ModuleList(
        #     (0): Linear(in_features=320, out_features=320, bias=True)
        #     (1): Dropout(p=0.0, inplace=False)
        #   )
        #   (processor): AnimationIDAttnNormalizedProcessor(
        #     (id_to_k): Linear(in_features=1024, out_features=320, bias=False)
        #     (id_to_v): Linear(in_features=1024, out_features=320, bias=False)
        #   )
        # )
        assert hidden_states is not None
        assert encoder_hidden_states is not None
        assert attention_mask == None
        assert temb == None
        assert scale == 1.0
        
        residual = hidden_states
        input_ndim = hidden_states.ndim

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        assert attention_mask == None
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        assert attention_mask == None

        query = attn.to_q(hidden_states)

        encoder_hidden_states = encoder_hidden_states.to(hidden_states.dtype)

        end_pos = encoder_hidden_states.shape[1] - self.num_tokens
        encoder_hidden_states, ip_hidden_states = (
            encoder_hidden_states[:, :end_pos, :],
            encoder_hidden_states[:, end_pos:, :],
        )

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = attn.head_to_batch_dim(query).contiguous()
        key = attn.head_to_batch_dim(key).contiguous()
        value = attn.head_to_batch_dim(value).contiguous()

        key = key.to(query.dtype)
        value = value.to(query.dtype)

        assert attention_mask == None
        if is_xformers_available(): # True
            ### xformers
            hidden_states = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=attention_mask)
            hidden_states = hidden_states.to(query.dtype)
        else:
            hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=0.0,
                                                           is_causal=False)
            hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.batch_to_head_dim(hidden_states)

        ip_key = self.id_to_k(ip_hidden_states)
        ip_value = self.id_to_v(ip_hidden_states)

        ip_key = attn.head_to_batch_dim(ip_key).contiguous()
        ip_value = attn.head_to_batch_dim(ip_value).contiguous()
        ip_key = ip_key.to(query.dtype)
        ip_value = ip_value.to(query.dtype)

        if is_xformers_available(): # True
            ### xformers
            ip_hidden_states = xformers.ops.memory_efficient_attention(query, ip_key, ip_value, attn_bias=None)
            ip_hidden_states = ip_hidden_states.to(query.dtype)
        else:
            ip_hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
                                                              is_causal=False)
            ip_hidden_states = ip_hidden_states.to(query.dtype)

        ip_hidden_states = attn.batch_to_head_dim(ip_hidden_states)
        mean_latents, std_latents = torch.mean(hidden_states, dim=(1, 2), keepdim=True), torch.std(hidden_states, dim=(1, 2), keepdim=True)
        mean_ip, std_ip = torch.mean(ip_hidden_states, dim=(1, 2), keepdim=True), torch.std(ip_hidden_states, dim=(1, 2), keepdim=True)
        ip_hidden_states = (ip_hidden_states - mean_ip) * (std_latents / (std_ip + 1e-5)) + mean_latents
        hidden_states = hidden_states + self.scale * ip_hidden_states
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states
