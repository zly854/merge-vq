# merge 后还原全部token，然后pixeldecoder



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


class VVQ(nn.Module):
    def __init__(self):    
        super().__init__()
        self.encoder = ViTEncoder(pretrained = False)
        self.encoder = apply_tome(self.encoder)
        self.quantizer = VectorQuant(feature_size = 192, num_codes = 1024)
        self.decoder = TiTokDecoder()
        self.layers = nn.ModuleList([
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(192, 96, kernel_size=3, stride=1, padding=1),
                nn.GELU(),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(96, 48, kernel_size=3, stride=1, padding=1),
                nn.GELU(),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(48, 16, kernel_size=3, stride=1, padding=1),
                nn.GELU(),
                nn.Upsample(scale_factor=2,mode = 'nearest'),
                nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1),
        ])


       
    def forward(self,x):
        x = self.encoder(x)
        #x = x[:, 1:, :]
        x = self.decoder(x)
        x = x.permute(0,2,1)
        z_q , vq_dict = self.quantizer(x)
        z_q = z_q.view(64,192,14,14)
        for layers in self.layers:
            z_q = layers(z_q)
        return z_q.clamp(-1, 1), vq_dict


data_dir = '/usr/data/zly/data/cifar10'


transform = transforms.Compose([
    transforms.Resize(224),  
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root=data_dir,
    train=True,
    download=False,  
    transform=transform
)


testset = torchvision.datasets.CIFAR10(
    root=data_dir,
    train=False,
    download=False,  
    transform=transform
)


trainloader = DataLoader(
    trainset,
    batch_size=64,
    shuffle=True,
    num_workers=2
)

testloader = DataLoader(
    testset,
    batch_size=64,
    shuffle=False,
    num_workers=2
)

def save_checkpoint(state, filename='checkpoint.pth'):
    torch.save(state, filename)



def train(model, train_loader, train_iterations=1000, num_epochs = 200, alpha=10):
    device = torch.device(f'cuda:{2}' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    def iterate_dataset(data_loader):
        data_iter = iter(data_loader)
        while True:
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(data_loader)
                x, y = next(data_iter)
            yield x.to(device), y.to(device)
    total_iterations = len(train_loader)
    total_time = 0

    for epoch in range(num_epochs):
        pbar = tqdm(range(total_iterations))
        for i in pbar:
            start_time = time.time()
            
            opt.zero_grad()
            x, _ = next(iterate_dataset(train_loader))
            out, vq_out = model(x)
            rec_loss = (out - x).abs().mean()
            cmt_loss = vq_out['loss']
            (rec_loss + alpha * cmt_loss).backward()

            opt.step()
            
            end_time = time.time()
            iteration_time = end_time - start_time
            total_time += iteration_time

            pbar.set_description(f'Epoch [{epoch+1}/{num_epochs}] rec loss: {rec_loss.item():.3f} | ' + \
                                 f'cmt loss: {cmt_loss.item():.3f} | ' + \
                                 f'active %: {vq_out["q"].unique().numel() / 1024 * 100:.3f}')

        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
        save_checkpoint({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'loss': rec_loss.item(),
            'total_time': total_time,
        }, filename=checkpoint_path)

    avg_time_per_iter = total_time / (num_epochs * total_iterations)
    print(f'Average time per iteration: {avg_time_per_iter:.3f}s')




    pbar = tqdm(range(train_iterations))
    for i in pbar:
        start_time = time.time()


        opt.zero_grad()
        x, _ = next(iterate_dataset(train_loader))
        out, vq_out = model(x)
        rec_loss = (out - x).abs().mean()
        cmt_loss = vq_out['loss']
        (rec_loss + alpha * cmt_loss).backward()

        opt.step()
        end_time = time.time()
        iteration_time = end_time - start_time
        total_time += iteration_time


        pbar.set_description(f'rec loss: {rec_loss.item():.3f} | ' + \
                             f'cmt loss: {cmt_loss.item():.3f} | ' + \
                             f'active %: {vq_out["q"].unique().numel() / 1024 * 100:.3f}')
        
        if (i + 1) % 100 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{i+1}.pth')
            save_checkpoint({
                'iteration': i + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'loss': rec_loss.item(),
                'total_time': total_time,
            }, filename=checkpoint_path)
        avg_time_per_iter = total_time / train_iterations
        print(f'Average time per iteration: {avg_time_per_iter:.3f}s')


    return

checkpoint_dir = '/usr/data/zly/checkpoints/merge_tiny_rec'
os.makedirs(checkpoint_dir, exist_ok=True)
torch.random.manual_seed(1324)
vvq = VVQ()
opt = torch.optim.AdamW(vvq.parameters(), lr=0.001)
train(vvq,trainloader)
