from bidirectional_transformer import BidirectionalTransformer
import torch
import torch.nn as nn
import argparse
import tome
import timm
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from timm.models.vision_transformer import VisionTransformer
from codebook import Codebook
from decoder import Decoder
from TiTok.blocks import TiTokDecoder
from TiTok.quantizer import VectorQuantizer
from maskgit_vqgan import VectorQuantizer as pixel_quantizer
from maskgit_vqgan import Decoder as pixel_decoder
from omegaconf import OmegaConf
import torch.optim as optim




# Config
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
args.batch_size = 16
args.accum_grad = 25
args.epochs = 16
args.start_from_epoch = 0
args.num_image_tokens = 256





# Baseline ViT base

class ViTEncoder(VisionTransformer):
    def __init__(self, model_name='vit_base_patch16_224', pretrained=True):
        model_cfg = timm.models.vision_transformer.default_cfgs[model_name]
        super().__init__(img_size=model_cfg['input_size'][1], patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, norm_layer=torch.nn.LayerNorm)
        if pretrained:
            state_dict = timm.create_model(model_name, pretrained=True).state_dict()
            self.load_state_dict(state_dict)

    def forward(self, x):
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_tokens, x), dim=1)
        else:
            x = torch.cat((cls_tokens, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        for blk in self.blocks:
            x = blk(x)

        return x


def apply_tome(model):
    tome.patch.timm(model)
    model.r = (16,-1.0)
    return model




class merge_vq(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoder = ViTEncoder(model_name = 'vit_base_patch16_224',pretrained = False)
        self.merged_encoder = apply_tome(self.encoder)
        #self.codebook = Codebook(args)
        #self.bidirectional_transformer = BidirectionalTransformer(args)
        #self.token_num = args.num_image_tokens
        #self.sos_token = args.num_codebook_vectors + 1 #一个整数token N作为起始，不和codebook中的index冲突。
        #self.mask_token_id = args.num_codebook_vectors #所有的mask token也被转成1024
        self.batch_size = 16
        self.merged_token_num = 10
        self.token_size = args.dim 
        self.quantizer = VectorQuantizer(codebook_size = 1024, token_size = 768)
        self.decoder = TiTokDecoder(args)
        self.pixel_quantizer = pixel_quantizer(num_embeddings=1024, embedding_dim=256, commitment_cost=0.25)
        self.pixel_decoder = pixel_decoder(OmegaConf.create(
            {"channel_mult": [1, 1, 2, 2, 4],
             "num_resolutions": 5,
             "dropout": 0.0,
             "hidden_channels": 128,
             "num_channels": 3,
             "num_res_blocks": 2,
             "resolution": 256,
             "z_channels": 256}))
        self.final_conv = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.final_pool = nn.AdaptiveAvgPool2d((224, 224)) 



    def forward(self,x):
        x = self.merged_encoder(x) #(10,768) x_k

        x = x.reshape(self.batch_size, self.token_size,self.merged_token_num,1) # x,x_k 
        z_q , _ = self.quantizer(x)
        z_q = z_q.permute(0,1,3,2)
        decoded_latent = self.decoder(z_q) # feature map 1024 \times 16 \times 16  from merged to 256 
        quantized_states = torch.einsum(
            'nchw,cd->ndhw', decoded_latent.softmax(1),
             self.pixel_quantizer.embedding.weight)
        decoded = self.pixel_decoder(quantized_states)
        decoded = self.final_conv(decoded)
        decoded = self.final_pool(decoded)
        return decoded




