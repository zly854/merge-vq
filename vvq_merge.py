# merge + quantization 后使用较少的token直接decoder回图像



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
        self.layers = nn.ModuleList([
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(192, 96, kernel_size=3, stride=1, padding=1),
                nn.GELU(),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(96, 48, kernel_size=3, stride=1, padding=1),
                nn.GELU(),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(48, 24, kernel_size=3, stride=1, padding=1),
                nn.GELU(),
                nn.Upsample(scale_factor=2,mode = 'nearest'),
                nn.Conv2d(24, 12, kernel_size=3, stride=1, padding=1),
                nn.GELU(),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(12, 6, kernel_size=3, stride=1, padding=1),
                nn.GELU(),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(6, 3, kernel_size=3, stride=1, padding=1),
                nn.GELU(),
                nn.Upsample(size = (224,224),mode = 'bilinear',align_corners = False)
        ])
    def forward(self,x):
        x = self.encoder(x)
        #x = x[:, 1:, :]
        x = x.permute(0,2,1)
        z_q , vq_dict = self.quantizer(x)
        z_q = z_q.unsqueeze(3)
        z_q = F.interpolate(z_q,size=(3,3),mode='bilinear',align_corners = False)
        #z_q = z_q.view(64,192,14,14)
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

checkpoint_dir = '/usr/data/zly/checkpoints/merge_tiny'
os.makedirs(checkpoint_dir, exist_ok=True)
torch.random.manual_seed(1324)
vvq = VVQ()
opt = torch.optim.AdamW(vvq.parameters(), lr=0.001)
train(vvq,trainloader)
