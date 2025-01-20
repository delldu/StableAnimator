import math
import torch
import torch.nn as nn
from diffusers.models.modeling_utils import ModelMixin
import pdb
import todos

def reshape_tensor(x, heads):
    bs, length, width = x.shape
    x = x.view(bs, length, heads, -1)
    x = x.transpose(1, 2)
    x = x.reshape(bs, heads, length, -1)
    return x

class PerceiverAttention(nn.Module):
    def __init__(self, *, dim=1024, dim_head=64, heads=16):
        super().__init__()

        self.scale = dim_head**-0.5
        self.dim_head = dim_head # 64
        self.heads = heads
        inner_dim = dim_head * heads # 1024
        assert inner_dim == 1024

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        # pdb.set_trace()

    def forward(self, x, latents):
        x = self.norm1(x)
        latents = self.norm2(latents)

        b, l, _ = latents.shape

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        # attention
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v

        out = out.permute(0, 2, 1, 3).reshape(b, l, -1)
        # pdb.set_trace()

        return self.to_out(out)
    
def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )

class FacePerceiver(torch.nn.Module):
    def __init__(
        self,
        dim=1024,
        depth=4,
        dim_head=64,
        heads=16,
        embedding_dim=1024,
        output_dim=1024,
        ff_mult=4,
    ):
        super().__init__()
        assert depth == 4

        self.proj_in = torch.nn.Linear(embedding_dim, dim)
        self.proj_out = torch.nn.Linear(dim, output_dim)
        self.norm_out = torch.nn.LayerNorm(output_dim)
        self.layers = torch.nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                torch.nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

    def forward(self, latents, x):
        x = self.proj_in(x)
        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents
        latents = self.proj_out(latents)
        return self.norm_out(latents)

class FusionFaceId(ModelMixin):
# class FusionFaceId(nn.Module):
    def __init__(self, 
        cross_attention_dim=1024, 
        id_embeddings_dim=512, 
        clip_embeddings_dim=1024, 
        num_tokens=4):
        super().__init__()

        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens

        self.proj = torch.nn.Sequential(
            torch.nn.Linear(id_embeddings_dim, id_embeddings_dim*2),
            torch.nn.GELU(),
            torch.nn.Linear(id_embeddings_dim*2, cross_attention_dim*num_tokens),
        )

        self.norm = torch.nn.LayerNorm(cross_attention_dim)

        self.fusion_model = FacePerceiver(
            dim=cross_attention_dim,
            depth=4,
            dim_head=64,
            heads=cross_attention_dim // 64, # 16
            embedding_dim=clip_embeddings_dim,
            output_dim=cross_attention_dim,
            ff_mult=4,
        )
        # pdb.set_trace()


    def forward(self, id_embeds, clip_embeds, shortcut=False, scale=1.0):
        # tensor [id_embeds] size: [1, 512], min: -3.173828, max: 2.986328, mean: 0.00991
        # tensor [clip_embeds] size: [1, 1, 1024], min: -5.863281, max: 6.507812, mean: 0.004285
        assert shortcut == False
        assert scale == 1.0

        x = self.proj(id_embeds)
        x = x.reshape(-1, self.num_tokens, self.cross_attention_dim)
        x = self.norm(x) 
        out = self.fusion_model(x, clip_embeds)
        if shortcut:
            out = x + scale * out

        # tensor [out] size: [1, 4, 1024], min: -14.492188, max: 14.453125, mean: 3.8e-05
        return out
