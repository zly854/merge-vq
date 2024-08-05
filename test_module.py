import torch
import torch.nn as nn 
import torch.nn.functional as F
import torchvision
from torchvision import datasets
from tqdm import tqdm
from vqtorch.nn import VectorQuant
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import ViTModel, ViTConfig
import timm
from timm.models.vision_transformer import VisionTransformer, _cfg
import os
import time
import tome
from collections import OrderedDict
import argparse


from functools import partial
from timm.models.layers import DropPath
from einops import rearrange, repeat
parser = argparse.ArgumentParser(description="VQGAN")
parser.add_argument('--run-name', type=str, default=None)
parser.add_argument('--image-size', type=int, default=256, help='Image height and width.)')
parser.add_argument('--image-channels', type=int, default=3, help='Number of channels of images.')
parser.add_argument('--dataset-path', type=str, default='./data', help='Path to data.')
parser.add_argument('--checkpoint-path', type=str, default='./checkpoints/last_ckpt.pt', help='Path to checkpoint.')
parser.add_argument('--device', type=str, default="cuda", help='Which device the training is on.')
parser.add_argument('--batch-size', type=int, default=10, help='Batch size for training.')
parser.add_argument('--accum-grad', type=int, default=10, help='Number for gradient accumulation.')
parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
parser.add_argument('--start-from-epoch', type=int, default=1, help='Number of epochs to train.')
parser.add_argument('--ckpt-interval', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate.')
parser.add_argument('--sos-token', type=int, default=1025, help='Start of Sentence token.')
parser.add_argument('--n-layers', type=int, default=24, help='Number of layers of transformer.')
parser.add_argument('--dim', type=int, default=768, help='Dimension of transformer.')
parser.add_argument('--hidden-dim', type=int, default=3072, help='Dimension of transformer.')
parser.add_argument('--num-image-tokens', type=int, default=256, help='Number of image tokens.')
parser.add_argument('--latent-dim', type=int, default=32, help='Latent dimension n_z.')
parser.add_argument('--num-codebook-vectors', type=int, default=8192, help='Number of codebook vectors.')
parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar.')
args = parser.parse_args()
args.num_codebook_vectors = 1000
args.n_layers = 24
args.latent_dim = 768
args.dim = 768
args.hidden_dim = 3072
args.batch_size = 4
args.accum_grad = 25
args.epochs = 200
args.start_from_epoch = 0
args.num_image_tokens = 256
args.lr = 0.001
class ResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model,
            n_head,
            mlp_ratio = 4.0,
            act_layer = nn.GELU,
            norm_layer = nn.LayerNorm
        ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.mlp_ratio = mlp_ratio
        # optionally we can disable the FFN
        if mlp_ratio > 0:
            self.ln_2 = norm_layer(d_model)
            mlp_width = int(d_model * mlp_ratio)
            self.mlp = nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(d_model, mlp_width)),
                ("gelu", act_layer()),
                ("c_proj", nn.Linear(mlp_width, d_model))
            ]))

    def attention(
            self,
            x: torch.Tensor
    ):
        return self.attn(x, x, x, need_weights=False)[0]

    def forward(
            self,
            x: torch.Tensor,
    ):
        attn_output = self.attention(x=self.ln_1(x))
        x = x + attn_output
        if self.mlp_ratio > 0:
            x = x + self.mlp(self.ln_2(x))
        return x


def _expand_token(token, batch_size: int):
    return token.unsqueeze(0).expand(batch_size, -1, -1)



class ViTEncoder(nn.Module):
    def __init__(self,model_name = "vit_tiny_patch16_224",pretrained = False):
        super(ViTEncoder,self).__init__()
        self.model = timm.create_model(model_name,pretrained)
        self.patch_embed = self.model.patch_embed
        self.cls_token = self.model.cls_token
        self.pos_embed = self.model.pos_embed
        self.pos_drop = self.model.pos_drop
        self.blocks = self.model.blocks
        self.norm = self.model.norm

    def forward(self, x):
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x


def apply_tome(model):
    tome.patch.timm(model)
    model.r = (16,-1.0)
    return model






class TiTokDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        #self.config = config
        self.image_size = 224
        self.patch_size = 16
        self.grid_size = self.image_size // self.patch_size
        self.model_size = "large"
        self.num_latent_tokens = 10
        self.token_size = 192
        self.width = {
                "small": 512,
                "base": 768,
                "large": 1024,
            }[self.model_size]
        self.num_layers = {
                "small": 8,
                "base": 12,
                "large": 24,
            }[self.model_size]
        self.num_heads = {
                "small": 8,
                "base": 12,
                "large": 16,
            }[self.model_size]

        self.decoder_embed = nn.Linear(
            self.token_size, self.width, bias=True)
        scale = self.width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(1, self.width))
        self.positional_embedding = nn.Parameter(
                scale * torch.randn(self.grid_size ** 2 + 1, self.width))
        # add mask token and query pos embed
        self.mask_token = nn.Parameter(scale * torch.randn(1, 1, self.width))
        self.latent_token_positional_embedding = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens, self.width))
        self.ln_pre = nn.LayerNorm(self.width)
        self.transformer = nn.ModuleList()
        for i in range(self.num_layers):
            self.transformer.append(ResidualAttentionBlock(
                self.width, self.num_heads, mlp_ratio=4.0
            ))
        self.ln_post = nn.LayerNorm(self.width)

        self.ffn = nn.Linear(1024,192)
        self.norm = nn.LayerNorm(192)
        self.conv_out = nn.Identity()
    
    def forward(self, x):
        #N, C, H, W = z_quantized.shape
        #assert H == 1 and W == self.num_latent_tokens, f"{H}, {W}, {self.num_latent_tokens}"
        #x = z_quantized.reshape(N, C*H, W).permute(0, 2, 1) # NLD

        x = self.decoder_embed(x)

        batchsize, seq_len, _ = x.shape

        mask_tokens = self.mask_token.repeat(batchsize, self.grid_size**2, 1).to(x.dtype)
        mask_tokens = torch.cat([_expand_token(self.class_embedding, mask_tokens.shape[0]).to(mask_tokens.dtype),
                                    mask_tokens], dim=1)
        mask_tokens = mask_tokens + self.positional_embedding.to(mask_tokens.dtype)
        x = x + self.latent_token_positional_embedding[:seq_len]
        x = torch.cat([mask_tokens, x], dim=1)

        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        for i in range(self.num_layers):
            x = self.transformer[i](x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = x[:, 1:1+self.grid_size**2] # remove cls embed
        x = self.ln_post(x)
        # N L D -> N D H W
        #x = x.permute(0, 2, 1).reshape(batchsize, self.width, self.grid_size, self.grid_size)
        x = self.ffn(x.contiguous())
        x = self.norm(x)
        #x = self.conv_out(x)
        return x
    


class Cross_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., length=27):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.linear_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.linear_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.linear_v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x_1, x_2, x_3):
        B, N, C = x_1.shape
        B, N_1, C = x_3.shape

        q = self.linear_q(x_1).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.linear_k(x_2).reshape(B, N_1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.linear_v(x_3).reshape(B, N_1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x




cross_att = Cross_Attention(dim = 192,num_heads=8,qkv_bias=True,attn_drop=0., proj_drop = 0.)
x_token = nn.Parameter(torch.zeros(1, 196, 192)) 
x_token = repeat(x_token,'() f c -> b f c', b = 64  )
print(x_token.shape)






encoder = ViTEncoder()
x = torch.rand(64,3,224,224)
encoder = apply_tome(encoder)
merged_tokens = encoder(x)
print(merged_tokens.shape)
#x_token = x_token.permute(0,2,1)
#merged_tokens = merged_tokens.permute(0,2,1)
print(x_token.shape)
merged_tokens = x_token + cross_att(x_token,merged_tokens,merged_tokens)
#decoder = TiTokDecoder()
#output = decoder(merged_tokens)
print(merged_tokens.shape)