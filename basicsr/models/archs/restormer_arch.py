## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881


import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse
from pdb import set_trace as stx
import numbers

from einops import rearrange



##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class DenseLayer(nn.Module):
    def __init__(self, nChannels, growthRate, kernel_size=1):
        super(DenseLayer, self).__init__()
        #self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                              #bias=False)
        self.conv = nn.Sequential(
            nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                              bias=False),nn.BatchNorm2d(growthRate)
        )
        self.bat = nn.BatchNorm2d(growthRate),
        self.leaky=nn.LeakyReLU(0.1,inplace=True)

    def forward(self, x):
        out = self.leaky(self.conv(x))
        out = torch.cat((x, out), 1)
        return out

##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x



##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out



##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x



##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x



##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=False, groups=n_feat),
                                  nn.Conv2d(n_feat, n_feat//2, kernel_size=1, stride=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=False, groups=n_feat),
                                  nn.Conv2d(n_feat, 2*n_feat, kernel_size=1, stride=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)
    
    
    
##########################################################################
##-------- Low Frequency Block----------------
class LowFrequencyBlock(nn.Module):
    def __init__(self,
        channels, 
        nDenselayer=1, 
        growthRate=16   
    ):
        super(LowFrequencyBlock, self).__init__()
        
        denseLayers_1 = []
        channels_1 = channels
        for i in range(nDenselayer):
            denseLayers_1.append(DenseLayer(channels_1, growthRate))
            channels_1 += growthRate
        denseLayers_1.append(nn.Conv2d(channels_1, channels, 1))
        self.magnitude_branch = nn.Sequential(*denseLayers_1)
        
        denseLayers_2 = []
        channels_2 = channels
        for i in range(nDenselayer):
            denseLayers_2.append(DenseLayer(channels_2, growthRate))
            channels_2 += growthRate
        denseLayers_2.append(nn.Conv2d(channels_2, channels, 1))
        self.phase_branch = nn.Sequential(*denseLayers_2)
    
    def forward(self, x):
        _, _, H, W = x.shape
        X_fft = torch.fft.fft2(x, norm='backward')
        
        magnitude = torch.abs(X_fft)
        log_magnitude = torch.log1p(magnitude)
        phase = torch.angle(X_fft) 
        
        log_magnitude = self.magnitude_branch(log_magnitude)
        magnitude = torch.expm1(log_magnitude)
        phase = self.phase_branch(phase)
        
        real = magnitude * torch.cos(phase)
        imag = magnitude * torch.sin(phase)
        
        x_out = torch.complex(real, imag)
        out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')
        out = out

        return out


##########################################################################
##-------- High Frequency Block----------------
class HighFrequencyBlock(nn.Module):
    def __init__(self,
        channels,     
        num_heads=8,
        bias=True
    ):
        super(HighFrequencyBlock, self).__init__()
        self.fuse_conv = nn.Sequential(*[
            nn.Conv2d(channels*3, channels, kernel_size=1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels),
            nn.Conv2d(channels, channels, kernel_size=1)
        ])
        self.attn = Attention(channels, num_heads, bias)
        self.split_conv  = nn.Sequential(*[
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels),
            nn.Conv2d(channels, channels*3, kernel_size=1)
        ])
    
    def forward(self, x):
        b, c, n, h, w = x.shape
        x = rearrange(x, 'b c n h w -> b (c n) h w')
        x = self.fuse_conv(x)
        x = x + self.attn(x)
        x = self.split_conv(x)
        out = rearrange(x, 'b (c n) h w -> b c n h w', n = n)  # đúng shape (B, C, 3, H, W)
        return out
    
class HighFrequencyBlock_02(nn.Module):
    def __init__(self,
        channels,     
        num_heads=8,
        bias=True
    ):
        super(HighFrequencyBlock_02, self).__init__()
        self.fuse_conv = nn.Sequential(*[
            nn.Conv2d(channels*3, channels, kernel_size=1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels),
            nn.Conv2d(channels, channels, kernel_size=1)
        ])
        self.attn = nn.Sequential(*[
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid()
        ])
        self.split_conv  = nn.Sequential(*[
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels),
            nn.Conv2d(channels, channels*3, kernel_size=1)
        ])
    
    def forward(self, x):
        b, c, n, h, w = x.shape
        x = rearrange(x, 'b c n h w -> b (c n) h w')
        x = self.fuse_conv(x)
        x = x * self.attn(x)
        x = self.split_conv(x)
        out = rearrange(x, 'b (c n) h w -> b c n h w', n = n)  # đúng shape (B, C, 3, H, W)
        return out
    
class DWTBlock(nn.Module):
    def __init__(self, channels, num_heads, LayerNorm_type, bias):
        super(DWTBlock, self).__init__()
        self.norm = LayerNorm(channels, LayerNorm_type)
        self.xfm = DWTForward(J=1, mode='zero', wave='haar')   # DWT
        self.ifm = DWTInverse(mode='zero', wave='haar')        # IDWT
        # self.high_branch = HighFrequencyBlock_02(channels, num_heads, bias)
        self.low_branch = LowFrequencyBlock(channels)
        self.prj_conv = nn.Conv2d(channels, channels, 1)
        
    
    def forward(self, x):
        x = self.norm(x)
        x_low, x_high  = self.xfm(x)
        # out_high = self.high_branch(x_high[0])
        out_low = self.low_branch(x_low)
        out = self.ifm((out_low, x_high))
        out = self.prj_conv(out)
        out = out + x
        return out

class SCFNBlock(nn.Module): # ref: Mishra_U-ENHANCE_Underwater_Image_Enhancement_Using_Wavelet_Triple_Self-Attention_ACCVW_2024_paper.pdf
    def __init__(self, channels, ffn_expansion_factor, LayerNorm_type):
        super(SCFNBlock, self).__init__()
        hidden_channels = int(channels * ffn_expansion_factor)
        self.norm = LayerNorm(channels, LayerNorm_type)
        self.prj_conv1 = nn.Conv2d(channels, hidden_channels, 1)
        self.conv = nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1, groups=hidden_channels)
        self.act = nn.GELU()
        self.prj_conv2 = nn.Conv2d(hidden_channels, channels, 1)
    
    def forward(self, x):
        out = self.norm(x)
        out = self.prj_conv1(out)
        out1 = self.conv(out)
        out = self.act(out1) * out
        out = self.prj_conv2(out) + x
        return out

class Block(nn.Module):
    def __init__(self, 
        channels, 
        num_heads,
        ffn_expansion_factor, 
        LayerNorm_type,
        bias
    ):
        super(Block, self).__init__()
        self.dwtblock = DWTBlock(channels, num_heads, LayerNorm_type, bias)
        self.refine = SCFNBlock(channels, ffn_expansion_factor, LayerNorm_type)
    
    def forward(self, x):
        out = self.dwtblock(x)
        out = self.refine(x)
        return out
    
class WFUWNet(nn.Module):
    def __init__(self, 
        inp_channels=3,
        out_channels=3,
        dim = 32,
        num_heads=[1,2,4,8],
        ffn_expansion_factor = 2,
        stages = 2,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
    ):
        super(WFUWNet, self).__init__()
        self.embed = OverlapPatchEmbed(inp_channels, dim, bias)
        self.recon = nn.Conv2d(dim, out_channels, 3, 1, 1)
        self.encoders = []
        self.decoders = []
        
        channels = dim
        for i in range(stages):
            encoder = [
                Block(channels, num_heads[i], ffn_expansion_factor, LayerNorm_type, bias),
                Downsample(channels)
            ]
            self.encoders += encoder
            channels = channels * 2
        
        self.middle = Block(channels, num_heads[-1], ffn_expansion_factor, LayerNorm_type, bias)
        
        for i in reversed(range(stages)):
            channels = channels // 2
            decoder = [
                Upsample(channels * 2),
                nn.Conv2d(channels * 2, channels, 1),
                Block(channels, num_heads[i], ffn_expansion_factor, LayerNorm_type, bias)
            ]
            self.decoders += decoder
        
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)
    
    def forward(self, x):
        out = self.embed(x)
        encodes = []
        for i in range(0, len(self.encoders), 2):
            out = self.encoders[i](out)
            encodes.append(out)
            out = self.encoders[i+1](out)
        
        out = self.middle(out)
        
        for i in range(0, len(self.decoders), 3):
            out = self.decoders[i](out)
            out = torch.concat([out, encodes[-1]], dim=1)
            out = self.decoders[i+1](out)
            out = self.decoders[i+2](out)
            encodes.pop()

        out = self.recon(out) + x
        
        return out
    
if __name__ == '__main__':
    img = torch.randn((1, 3, 512, 512))

    model = WFUWNet()
    print(model(img).shape)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Tổng số tham số: {total_params}')
