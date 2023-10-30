import torch
import torch
from torch import nn
import numpy as np
import pytorch_lightning as pl
from torchbench.datasets import ADE20K
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_tensor, pil_to_tensor, resize, normalize
from torchvision.transforms import InterpolationMode
from torch.utils.data import Subset, DataLoader
from tqdm import tqdm
# Combine adjacent patches to reduce feature map size, followed by LayerNorm
# Hint, patch merging can be accomplished with a single Conv2D module
class OverlapPatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size, stride, padding):
        super().__init__()
        #CNNs are used in overlap merging for the segformer. Because they can merge information across layers
        self.cn1 = nn.Conv2d(in_channels, out_channels, kernel_size=patch_size, stride = stride, padding = padding)
        self.layerNorm = nn.LayerNorm(out_channels)

    def forward(self, patches):
        """Merge patches to reduce dimensions of input.

        :param patches: tensor with shape (B, C, H, W) where
            B is the Batch size
            C is the number of Channels
            H and W are the Height and Width
        :return: tensor with shape (B, C, H, W)
        """
        #Patches = B, C, H, W
        #@TODO: Simulate and get the shapes
        x = self.cn1(patches)
        _,_,H, W = x.shape
        x = x.flatten(2).transpose(1,2) #Flatten - (B,C,H*W); transpose B,HW, C
        x = self.layerNorm(x)
        #Later in the stupid paper, they call HW = N
        #WE also denote C as embed dimension from now
        return x,H,W #B, N, EmbedDim
    
class EfficientSelfAttention(nn.Module):
    def __init__(self, channels, reduction_ratio, num_heads):
        super().__init__()
        assert channels % num_heads == 0, f"channels {channels} should be divided by num_heads {num_heads}."

        self.heads= num_heads

        #### Self Attention Block consists of 2 parts - Reduction and then normal Attention equation of queries and keys###
        
        # Reduction Parameters #
        self.cn1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=reduction_ratio, stride= reduction_ratio)
        self.ln1 = nn.LayerNorm(channels)
        # Attention Parameters #
        self.keyValueExtractor = nn.Linear(channels, channels * 2)
        self.query = nn.Linear(channels, channels)
        self.smax = nn.Softmax(dim=-1)
        self.finalLayer = nn.Linear(channels, channels) #Projection Layer


        
    #@NOTE - Added H and W to the input since we need to modify the shape
    def forward(self, x, H, W):

        """ Perform self attention with reduced sequence length

        :param x: tensor of shape (B, N, C) where
            B is the batch size,
            N is the number of queries (equal to H * W)
            C is the number of channels
        :return: tensor of shape (B, N, C)
        """
        B,N,C = x.shape
        # First lets do reduction
        # B, N, C -> B, C, N
        x1 = x.clone().permute(0,2,1)
        # BCN -> BCHW
        x1 = x1.reshape(B,C,H,W)
        # BCHW -> (Conv) -> BC, H/r,W/r
        x1 = self.cn1(x1)
        # BC, H/r, W/r -> B,C,HW/r -> B, HW/r, C
        x1 = x1.reshape(B,C,-1).permute(0,2,1)
        # Layer norm as in pyramid net
        x1 = self.ln1(x1)
        # We have got the Reduced Embeddings! We need to extract key and value pairs now
        # Linear transform to X -> XW+b (bnc -> bn2c)
        keyVal = self.keyValueExtractor(x1)
        # bn2c-> 2,b,heads, n, c/heads
        keyVal = keyVal.reshape(B, -1 , 2, self.heads, int(C/self.heads)).permute(2,0,3,1,4)
        #Extract Key and Value
        k,v = keyVal[0],keyVal[1] #b,heads, n, c/heads

        #Query -> b, heads, n, c/heads
        q = self.query(x).reshape(B, N, self.heads, int(C/self.heads)).permute(0, 2, 1, 3)

        #Finally we can do the attention thing
        dimHead = (C/self.heads)**0.5
        attention = self.smax(q@k.transpose(-2, -1)/dimHead)
        attention = (attention@v).transpose(1,2).reshape(B,N,C)

        x = self.finalLayer(attention) #B,N,C        
        return x
class MixFFN(nn.Module):
    def __init__(self, channels, expansion_factor):
        super().__init__()
        ### @WARNING: We are not doing skip connection inside this function, carry forward to the next fn, although given in the paper
        expanded_channels = channels*expansion_factor
        #Instead of the standard ViT MLP Layers, we use Convs to preserve spatial information for the Segformer architecture - wild
        #MLP Layer        
        self.mlp1 = nn.Linear(channels, expanded_channels)
        #Depth Wise CNN Layer
        self.depthwise = nn.Conv2d(expanded_channels, expanded_channels, kernel_size=3,  padding='same', groups=channels)
        #GELU
        self.gelu = nn.GELU()
        #MLP to predict
        self.mlp2 = nn.Linear(expanded_channels, channels)

    def forward(self, x, H, W):
        """ Perform self attention with reduced sequence length

        :param x: tensor with shape (B, C, H, W) where
            B is the Batch size
            C is the number of Channels
            H and W are the Height and Width
        :return: tensor with shape (B, C, H, W)
        """
        # Input BNC instead of BCHW
        # BNC -> B,N,C*exp 
        x = self.mlp1(x)
        B,N,C = x.shape
        # Prepare for the CNN operation, channel should be 1st dim
        # B,N, C*exp -> B, C*exp, H, W 
        x = x.transpose(1,2).view(B,C,H,W)

        #Depth Conv - B, N, Cexp 
        x = self.gelu(self.depthwise(x).flatten(2).transpose(1,2))

        #Back to the orignal shape
        x = self.mlp2(x) # BNC
        return x
class MixTransformerEncoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size, stride, padding, 
                 n_layers, reduction_ratio, num_heads, expansion_factor):
        super().__init__()
        self.patchMerge = OverlapPatchMerging(in_channels, out_channels, patch_size, stride, padding) # B N embed dim
        #You might be wondering why I didn't used a cleaner implementation but the input to each forward function is different
        self._attn = nn.ModuleList([EfficientSelfAttention(out_channels, reduction_ratio, num_heads) for _ in range(n_layers)])
        self._ffn = nn.ModuleList([MixFFN(out_channels,expansion_factor) for _ in range(n_layers)])
        self._lNorm = nn.ModuleList([nn.LayerNorm(out_channels) for _ in range(n_layers)])

    def forward(self, x):
        """ Run one block of the mix vision transformer

        :param x: tensor with shape (B, C, H, W) where
            B is the Batch size
            C is the number of Channels
            H and W are the Height and Width
        :return: tensor with shape (B, C, H, W)
        """
        #Extract orignal shape
        B,C,H,W = x.shape
        x,H,W = self.patchMerge(x) # B N embed dim (C)
        for i in range(len(self._attn)):
            x = x + self._attn[i].forward(x, H, W) #BNC
            x = x + self._ffn[i].forward(x, H, W) #BNC
            x = self._lNorm[i].forward(x) #BNC
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() #BCHW
        return x

from torch.nn.functional import interpolate
class MLPDecoder(nn.Module):
    def __init__(self, in_channels, embed_channels, out_dims, num_classes):
        """Initialize MLP Decoder.
        :param in_channels: list of int, number of input channels for each layeer
        :param embed_channels: int, common embedding channels
        :param out_dims: tuple, dimensions of common feature map
        """
        super().__init__()
        m = len(in_channels) #Assuming this is what is meant by m
        self.outDim = out_dims #Not sure where to use this
        #MLPi to get all features to common dim - I will use convs because linear seems funny on a 3D tensor
        self.mlp1 = nn.ModuleList([nn.Conv2d(in_channels[i],embed_channels, kernel_size=1) for i in range(m)])        
        self.mlp2 = nn.Conv2d(m*embed_channels, embed_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(embed_channels)
        self.relu = nn.ReLU(inplace=True)
        self.mlp3 = nn.Conv2d(embed_channels, num_classes, kernel_size=1)


    def forward(self, x):
        """ Take feature map to common shape
        :param x: list of tensors with shape (B, Ci, Hi, Wi)
            B is the Batch size
            Ci is the number of input Channels
            Hi and Wi are the input Height and Width
        :return: tensor with shape (B, C, H, W)
            B is the Batch size
            C is the number of output Channels
            H and W are the input Height and Width
        """
        # First Pass it through the MLP_i
        for i in range(len(x)):
            x[i] = self.mlp1[i].forward(x[i])
            x[i] = interpolate(x[i], size=self.outDim, mode='bilinear')
        #Concatenate
        x = torch.cat(x, dim = 1)
        x = self.relu(self.bn(self.mlp2(x)))
        x = self.mlp3(x) 


        
        return x
class SegFormer_B0(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder_blocks = nn.ModuleList([
            MixTransformerEncoderLayer(3, 32, patch_size=7, stride=4, padding=3, n_layers=2, reduction_ratio=8, num_heads=1, expansion_factor=8),
            MixTransformerEncoderLayer(32, 64, patch_size=3, stride=2, padding=1, n_layers=2, reduction_ratio=4, num_heads=2, expansion_factor=8),
            MixTransformerEncoderLayer(64, 160, patch_size=3, stride=2, padding=1, n_layers=2, reduction_ratio=2, num_heads=5, expansion_factor=4),
            MixTransformerEncoderLayer(160, 256, patch_size=3, stride=2, padding=1, n_layers=2, reduction_ratio=1, num_heads=8, expansion_factor=4)
        ])
        self.decoder = MLPDecoder([32, 64, 160, 256], 256, (64, 64), 4)
        self.loss = nn.CrossEntropyLoss()
    
    def forward(self, images):
        embeds = [images]
        for block in self.encoder_blocks:
            embeds.append(block(embeds[-1]))
        return self.decoder(embeds[1:])
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=6e-5)
        return optimizer
    
    def miou(self, prediction, targets):
        return 0

    def training_step(self, batch, batch_idx):
        images, targets = batch
        predictions = self.forward(images)
        loss = self.loss(predictions, targets)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        return
        images, targets = batch
        predictions = self.forward(images)
        miou = self.miou(predictions, targets)
        self.log('miou', miou, prog_bar=True)
model = SegFormer_B0()
model = model.load_from_checkpoint(checkpoint_path=r"lightning_logs\version_14\checkpoints\epoch=8-step=3114.ckpt",map_location='cpu')
print(model.logger)
