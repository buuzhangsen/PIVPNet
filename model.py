import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.init import _calculate_fan_in_and_fan_out
from timm.models.layers import to_2tuple, trunc_normal_
# from .norm_layer import *

class ConvLayer(nn.Module):
    def __init__(self, net_depth, dim, kernel_size=3, gate_act=nn.Sigmoid):
        super().__init__()
        self.dim = dim

        self.net_depth = net_depth
        self.kernel_size = kernel_size

        self.Wv = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim, padding_mode='reflect')
        )

        self.Wg = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            gate_act() if gate_act in [nn.Sigmoid, nn.Tanh] else gate_act(inplace=True)
        )

        self.proj = nn.Conv2d(dim, dim, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            gain = (8 * self.net_depth) ** (-1/4)    # self.net_depth ** (-1/2), the deviation seems to be too small, a bigger one may be better
            fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            trunc_normal_(m.weight, std=std)

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, X):
        out = self.Wv(X) * self.Wg(X)
        out = self.proj(out)
        return out


class BasicBlock(nn.Module):
    def __init__(self, net_depth, dim, kernel_size=3, conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid):
        super().__init__()
        self.norm = norm_layer(dim)
        self.conv = conv_layer(net_depth, dim, kernel_size, gate_act)

    def forward(self, x):
        identity = x
        x = self.norm(x)
        x = self.conv(x)
        x = identity + x
        return x


class BasicLayer(nn.Module):
    def __init__(self, net_depth, dim, depth, kernel_size=3, conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid):
        super().__init__()
        self.dim = dim
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList([
            BasicBlock(net_depth, dim, kernel_size, conv_layer, norm_layer, gate_act)
            for i in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = patch_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size,
                              padding=(kernel_size-patch_size+1)//2, padding_mode='reflect')

    def forward(self, x):
        x = self.proj(x)
        return x


class PatchUnEmbed(nn.Module):
    def __init__(self, patch_size=4, out_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.out_chans = out_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = 1

        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans*patch_size**2, kernel_size=kernel_size,
                      padding=kernel_size//2, padding_mode='reflect'),
            nn.PixelShuffle(patch_size)
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class SKFusion(nn.Module):
    def __init__(self, dim, height=2, reduction=8):
        super(SKFusion, self).__init__()

        self.height = height
        d = max(int(dim/reduction), 4)

        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, d, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(d, dim*height, 1, bias=False)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_feats):
        B, C, H, W = in_feats[0].shape

        in_feats = torch.cat(in_feats, dim=1)
        in_feats = in_feats.view(B, self.height, C, H, W)

        feats_sum = torch.sum(in_feats, dim=1)
        attn = self.mlp(feats_sum)
        attn = self.softmax(attn.view(B, self.height, C, 1, 1))

        out = torch.sum(in_feats*attn, dim=1)
        return out


class LowRankLinear(nn.Module):
    def __init__(self, in_features, out_features, rank=16):
        super(LowRankLinear, self).__init__()
        self.U = nn.Parameter(torch.randn(out_features, rank))
        self.V = nn.Parameter(torch.randn(rank, in_features))

    def forward(self, x):
        return F.linear(x, torch.matmul(self.U, self.V))


class PromptGenBlock(nn.Module):
    def __init__(self, prompt_dim=128, prompt_len=7, prompt_size=96, lin_dim=64):
        super(PromptGenBlock, self).__init__()
        self.masks = torch.ones(prompt_len, prompt_dim, prompt_size, prompt_size)  # 初始化为 1x1 的尺寸
        
        self.conv1x1 = nn.Conv2d(prompt_dim, prompt_dim, kernel_size=1, stride=1, bias=False)
        self.linear_layer = nn.Linear(prompt_dim, prompt_len)
        
        # 将 beta 初始化为 (1, prompt_dim, 1, 1)，后续动态调整到输入尺寸
        self.beta = nn.Parameter(torch.ones(1, prompt_dim, prompt_size, prompt_size))  # 修改为 1x1 的基准尺寸

    def forward(self, x):
        B, C, H, W = x.shape
        
        # 生成 beta 的动态尺寸版本
        beta_resized = F.interpolate(self.beta, size=(H, W), mode="bilinear", align_corners=False)
        
        emb = x.mean(dim=(-2, -1))
        prompt_weights = F.softmax(self.linear_layer(emb), dim=1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        masks = self.masks.to(prompt_weights.device)
        
        # 动态调整 masks 的尺寸到输入大小
        masks_resized = F.interpolate(masks, size=(H, W), mode="bilinear", align_corners=False)
        
        # 组合 prompts
        combined_prompt = sum(weight * mask for weight, mask in zip(prompt_weights.split(1, dim=1), masks_resized)).squeeze(1)
        
        # 生成最终 prompt
        prompt = self.conv1x1(combined_prompt)
        
        # 动态调整后的 beta 参与计算
        output = prompt * x * beta_resized + x
        
        return output

class FeedForward(nn.Module):
    def __init__(self, dim, expand=0.85, bias=False):
        super(FeedForward, self).__init__()
        hidden_features = int(dim*expand)
        # groups = int()
        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)
        self.dwconv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=5, stride=1, padding=2, groups=hidden_features, bias=bias)
        self.dwconv3 = nn.Conv2d(dim, 2, kernel_size=3, padding=1, bias=bias)
        self.dwconv4 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
        self.act = nn.Sigmoid()

    def forward(self, x_in,illum):
        x = self.project_in(x_in)
        attn1 = self.dwconv(x) 
        attn2 = self.dwconv2(attn1)
        illum1,illum2 = self.dwconv3(illum).chunk(2, dim=1)
        attn = attn1*self.act(illum1)+attn2*self.act(illum2)
        x = x + attn*x
        x = F.gelu(self.dwconv4(x))
        x = self.project_out(x)
        return x
    
    

class gUNet(nn.Module):
    def __init__(self, kernel_size=5, base_dim=32, depths=[8, 8, 8, 16, 8, 8, 8], conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid, fusion_layer=SKFusion, PromptGenBlock=PromptGenBlock,FeedForward = FeedForward):
        super(gUNet, self).__init__()
        # setting
        assert len(depths) % 2 == 1
        stage_num = len(depths)
        half_num = stage_num // 2
        net_depth = sum(depths)
        embed_dims = [2**i*base_dim for i in range(half_num)]
        embed_dims = embed_dims + [2**half_num*base_dim] + embed_dims[::-1]

        self.patch_size = 2 ** (stage_num // 2)
        self.stage_num = stage_num
        self.half_num = half_num

        # input convolution
        self.inconv = PatchEmbed(patch_size=1, in_chans=3, embed_dim=embed_dims[0], kernel_size=3)

        # backbone
        self.layers = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.skips = nn.ModuleList()
        self.fusions = nn.ModuleList()
        self.prompt_gene = nn.ModuleList()
        self.prompt_layers_ = nn.ModuleList()
        self.fusion_conv = nn.ModuleList()
        self.FeedForward = nn.ModuleList()
        self.cat_x = nn.ModuleList()
        # self.pool_avg = nn.ModuleList()

        for i in range(self.stage_num):
            self.layers.append(BasicLayer(dim=embed_dims[i], depth=depths[i], net_depth=net_depth, kernel_size=kernel_size, 
                                          conv_layer=conv_layer, norm_layer=norm_layer, gate_act=gate_act))

        for i in range(self.half_num):
            self.downs.append(PatchEmbed(patch_size=2, in_chans=embed_dims[i], embed_dim=embed_dims[i+1]))
            self.ups.append(PatchUnEmbed(patch_size=2, out_chans=embed_dims[i], embed_dim=embed_dims[i+1]))
            self.skips.append(nn.Conv2d(embed_dims[i], embed_dims[i], 1))
            self.fusions.append(fusion_layer(embed_dims[i]))
            self.prompt_layers_.append(BasicLayer(dim=embed_dims[i+1], depth=4, net_depth=net_depth, kernel_size=kernel_size, 
                                                  conv_layer=conv_layer, norm_layer=norm_layer, gate_act=gate_act))
            self.prompt_gene.append(PromptGenBlock(prompt_dim=embed_dims[i+1], prompt_len=7, prompt_size=int(128/2**(i+1)), lin_dim=embed_dims[i+1]))
            self.fusion_conv.append(nn.Conv2d(embed_dims[i+1]*2, embed_dims[i+1], 1))
            self.FeedForward.append(FeedForward(dim=embed_dims[i+1]))
            self.cat_x.append(nn.Conv2d(5, embed_dims[i+1], 3,1,1))
            # self.pool_avg.append()

        # output convolution
        self.outconv = PatchUnEmbed(patch_size=1, out_chans=3, embed_dim=embed_dims[-1], kernel_size=3)

    def forward(self, x):
        x_max = torch.max(x, dim=1, keepdim=True)[0]
        x_mean = torch.mean(x, dim=1, keepdim=True)
        x_in_cat = torch.cat((x,x_max,x_mean), dim=1)
        feat = self.inconv(x)
        # x_2 = F.avg_pool2d(x_in_cat, kernel_size=2, stride=2)
        # x_4 = F.avg_pool2d(x_in_cat, kernel_size=4, stride=4)

        skips = []

        for i in range(self.half_num):
            feat = self.layers[i](feat)
            skips.append(self.skips[i](feat)) 
            feat = self.downs[i](feat)

        feat = self.layers[self.half_num](feat)

        for i in range(self.half_num-1, -1, -1):
            x_in = F.avg_pool2d(x_in_cat, kernel_size=(2**i)*2, stride=(2**i)*2)
            x_in = self.cat_x[i](x_in)
            prompt_x = self.FeedForward[i](feat,x_in)
            prompt_g = self.prompt_gene[i](feat)  # Correctly call each PromptGenBlock
            prompt_feat = torch.cat([prompt_g, feat], dim=1)
            prompt = self.fusion_conv[i](prompt_feat)

            feat = self.prompt_layers_[i](prompt)
            feat = self.ups[i](feat)
            feat = self.fusions[i]([feat, skips[i]])  
            feat = self.layers[self.stage_num-i-1](feat)

        x = self.outconv(feat) + x

        return x
