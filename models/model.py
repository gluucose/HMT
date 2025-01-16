import math
import numpy as np
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self,
                 channel_num: int = 16,
                 modal_num: int = 2,
                 is_cross: bool = True,
                 is_acmix: bool = True,
                 is_cnn: bool = False,
                 is_att: bool = False,
                 is_tabular: bool = True,
                 t_dim: int = 0,
                 bottleneck_factor: int = 7,
                 is_seg: bool = True,
                 is_ag: bool = True,
                 is_tatm: bool = True,
                 is_ham: bool = True):
        super().__init__()
        
        self.is_seg = is_seg
        self.is_tatm = is_tatm
        self.is_tabular = is_tabular
        is_acmix = is_ham
        is_cnn = not is_ham
        is_cross = is_ham
        
        self.encoder = Encoder(channel_num=channel_num//2,
                               modal_num=modal_num,
                               is_cross=is_cross,
                               is_acmix=is_acmix,
                               is_cnn=is_cnn,
                               is_att=is_att)
        
        if is_seg:
            self.decoder = Decoder(in_channels=channel_num * modal_num // 2,
                                channel_num=channel_num * modal_num // 4,
                                is_ag=is_ag)
            self.surv_head = SurvHead(channel_num=channel_num * modal_num // 4,
                                      interval_num=10,
                                      is_seg=is_seg,
                                      t_dim=t_dim if not is_tatm else 0)
        else:
            self.surv_head = SurvHead(channel_num=channel_num * modal_num,
                                      interval_num=10,
                                      is_seg=is_seg,
                                      t_dim=t_dim if not is_tatm else 0)
        
        if is_tatm:
            self.global_pool = nn.AdaptiveAvgPool3d(1)
            
            self.mlp = nn.ModuleList()
            self.i_dim = [(channel_num * modal_num // 2) * (2 ** i) for i in range(5)]
            
            for i in range(5):
                fusion_channel = self.i_dim[i] + t_dim
                bottleneck_channel = (self.i_dim[i] + t_dim) // bottleneck_factor
                self.mlp.append(nn.Sequential(
                    nn.Linear(fusion_channel, bottleneck_channel, bias=False),
                    nn.ReLU(),
                    nn.Linear(bottleneck_channel, 2 * self.i_dim[i], bias=False)
                ))
            
    def fuse(self, x, t, layer_num):
        if self.is_tatm:
            z = self.global_pool(x)
            z = z.view(z.size(0), -1)
            z = torch.cat((z, t), dim=1)
            z = self.mlp[layer_num](z)
            alpha, beta = torch.split(z, self.i_dim[layer_num], dim=1)
            alpha = alpha.view(*alpha.size(), 1, 1, 1).expand_as(x)
            beta = beta.view(*beta.size(), 1, 1, 1).expand_as(x)
            return (alpha * x) + beta
        else:
            return x
        
    def forward(self, x1, x2=None, t=None):
        x_features = self.encoder(x1, x2)
        
        if self.is_tatm:
            for i in range(5):
                x_features[i] = self.fuse(x_features[i], t, i)
        
        if self.is_seg:
            seg_pred, surv_feats, masks = self.decoder(x_features)
            surv_pred, reg_weight, feats = self.surv_head(surv_feats, t if not self.is_tatm else None)
        else:
            seg_pred, masks = None, None
            surv_pred, reg_weight, feats = self.surv_head([x_features[-1]], t)
            
        return seg_pred, surv_pred, reg_weight, feats, masks

#-------------------------------------------------------------------------------------- 

class Encoder(nn.Module):
    def __init__(self, 
                 channel_num: int, 
                 modal_num: int = 2,
                 is_cross: bool = True,
                 is_acmix: bool = True,
                 is_cnn: bool = False,
                 is_att: bool = False,):
        super().__init__()
        
        conv_num = [3, 3, 4, 4]
        num_layers = 2
        window_size=[5, 5, 5]
        is_cross_list = [False, is_cross, is_cross, is_cross]
        
        self.modal_num = modal_num
        self.encoder = nn.ModuleList([nn.ModuleList()])
        
        for i in range(modal_num):
            self.encoder.append(nn.ModuleList())
            self.encoder[i].append(ResidualBlock(in_channels=1,
                                                 out_channels=channel_num,
                                                 conv_num=2))
            for j in range(4):
                self.encoder[i].append(HybridBlock(in_channels=channel_num * (2 ** j),
                                                   embed_dim=channel_num * (2 ** (j+1)),
                                                   conv_num=conv_num[j],
                                                   num_layers=num_layers,
                                                   num_heads=int(channel_num * (2 ** (j-2))),
                                                   window_size=window_size,
                                                   is_cross=is_cross_list[j],
                                                   is_acmix=is_acmix,
                                                   is_cnn=is_cnn,
                                                   is_att=is_att,))
                
        self.downsample = nn.ModuleList()
        for i in range(4):
            self.downsample.append(nn.MaxPool3d(2, stride=2))
        
    def forward(self, x1, x2=None):
        if self.modal_num == 2:
            x_features = [None, None, None, None, None]
            x1_features = [None, None, None, None, None]
            x2_features = [None, None, None, None, None]
            for i in range(5):
                x1_features[i] = self.encoder[0][i](x1, x2)
                x2_features[i] = self.encoder[1][i](x2, x1)
                if i < 4:
                    x1 = self.downsample[i](x1_features[i])
                    x2 = self.downsample[i](x2_features[i])
            
            for i in range(5):
                x_features[i] = torch.cat([x1_features[i], x2_features[i]], dim=1)
        
        else:
            x_features = [None, None, None, None, None]
            for i in range(5):
                x_features[i] = self.encoder[0][i](x1)
                if i < 4:
                    x1 = self.downsample[i](x_features[i])
        
        return x_features

    
class Decoder(nn.Module):
    def __init__(self, 
                 in_channels: int,
                 channel_num: int,
                 is_ag: bool):
        super().__init__()
        
        self.is_ag = is_ag
        
        conv_num = [4, 3, 3, 2]
        
        self.decoder = nn.ModuleList([ResidualBlock(in_channels=in_channels*16,
                                                    out_channels=channel_num*16,
                                                    conv_num=4)])
        for i in range(4):
            self.decoder.append(ResidualBlock(in_channels=channel_num * (16 // (2 ** i)) + in_channels * (8 // (2 ** i)),
                                              out_channels=channel_num * (8 // (2 ** i)),
                                              conv_num=conv_num[i]))
        
        self.upsample = nn.ModuleList()
        for i in range(4):
            self.upsample.append(nn.Upsample(scale_factor=2, mode='nearest'))
            
        if is_ag:
            self.atten_gate = nn.ModuleList()
            for i in range(4):
                self.atten_gate.append(RegionAttenBlock(in_channels * (8 // (2 ** i)),
                                                        channel_num * (16 // (2 ** i)),
                                                        channel_num * (8 // (2 ** i))))
        
        self.conv = nn.Conv3d(channel_num, 1, kernel_size=1, stride=1, padding='same')
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x_features):
        x_de = [self.decoder[0](x_features[4])]
        masks = []
        
        if self.is_ag:
            for i in range(4):
                x_gate, mask = self.atten_gate[i](x_features[3-i], x_de[i])
                masks.append(mask)
                x_up = self.upsample[i](x_de[i])
                x = torch.cat([x_gate, x_up], dim=1)
                x_de.append(self.decoder[i+1](x))
        else:
            for i in range(4):
                x_up = self.upsample[i](x_de[i])
                x = torch.cat([x_up, x_up], dim=1)
                x_de.append(self.decoder[i+1](x))
        
        x = self.conv(x_de[4])
        seg_pred = self.sigmoid(x)
        surv_feats = x_de[1:]
        
        return seg_pred, surv_feats, masks

    
class SurvHead(nn.Module):
    def __init__(self, 
                 channel_num: int, 
                 interval_num: int,
                 is_seg: bool,
                 t_dim: int):
        super().__init__()
        
        self.dropout_1 = nn.Dropout(0.5)
        self.dropout_2 = nn.Dropout(0.5)
        
        self.dense_1 = nn.Linear(channel_num*(15 if is_seg else 8) + t_dim, channel_num*8)
        self.dense_2 = nn.Linear(channel_num*8, interval_num)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x_in, t=None):
        x_GAP = []
        for x in x_in:
            x = torch.mean(x, dim=(2,3,4))
            x_GAP.append(x)
        if t is not None:
            x_GAP.append(t)
        feats = torch.cat(x_GAP, dim=1)
        
        x = self.dropout_1(feats)
        x = self.dense_1(x)
        x = self.relu(x)
        
        x = self.dropout_2(x)
        x = self.dense_2(x)
        surv_pred = self.sigmoid(x)
        
        reg_weight = [self.dense_1.weight, self.dense_2.weight]
            
        return surv_pred, reg_weight, feats
    
    
#-------------------------------------------------------------------------------------- 
class HybridBlock(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 embed_dim: int, 
                 conv_num: int,
                 num_layers: int,
                 num_heads: int,
                 window_size: list,
                 is_cross: bool = True,
                 is_acmix: bool = True,
                 is_cnn: bool = False,
                 is_att: bool = False,):
        super().__init__()
        
        self.is_cross = is_cross
        self.is_acmix = is_acmix
        self.is_cnn = is_cnn
        self.is_att = is_att
        
        if (not is_acmix) and is_cnn:
            self.residual_block = ResidualBlock(in_channels=in_channels,
                                                out_channels=embed_dim,
                                                conv_num=conv_num)
        
        if is_acmix or is_att:
            self.proj_1 = nn.Conv3d(in_channels, embed_dim, kernel_size=1, stride=1)
            if is_cross:
                self.proj_2 = nn.Conv3d(in_channels, embed_dim, kernel_size=1, stride=1)
            self.swin_block = SwinTransStageBlock(embed_dim=embed_dim,
                                                  num_layers=num_layers,
                                                  num_heads=num_heads,
                                                  window_size=window_size,
                                                  cross_atten=is_cross,
                                                  is_acmix=is_acmix,)
    
    def forward(self, x_1, x_2=None):
        x_conv = 0
        x_trans = 0
        
        if (not self.is_acmix) and self.is_cnn:
            x_conv = self.residual_block(x_1)
        
        if self.is_acmix or self.is_att:
            x_1 = self.proj_1(x_1)
            if self.is_cross:
                x_2 = self.proj_2(x_2)
            x_trans = self.swin_block(x_1, x_2 if self.is_cross else None)
        
        x_out = x_conv + x_trans
        return x_out

    
class RegionAttenBlock(nn.Module):
    def __init__(self, channel_x, channel_g, channel_num):
        super().__init__()
        
        self.Conv_g = nn.Conv3d(channel_g, channel_num, kernel_size=1, stride=1, padding='same')
        self.BN_g = nn.BatchNorm3d(channel_num)
        
        self.Conv_x = nn.Conv3d(channel_x, channel_num, kernel_size=1, stride=1, padding='same')
        self.BN_x = nn.BatchNorm3d(channel_num)
        
        self.Conv_relu = nn.Conv3d(channel_num, 2, kernel_size=1, stride=1, padding='same')
        self.BN_relu = nn.BatchNorm3d(2)
        
        self.ReLU = nn.ReLU()
        self.Softmax = nn.Softmax(dim=1)
        
        self.AvgPool = nn.AvgPool3d(2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        
    def forward(self, x_in, g_in):
        g = self.Conv_g(g_in)
        g_int = self.BN_g(g)
        
        x = self.Conv_x(x_in)
        x = self.BN_x(x)
        x_int = self.AvgPool(x)
        
        x = torch.add(x_int, g_int)
        x_relu = self.ReLU(x)
        
        x = self.Conv_relu(x_relu)
        x = self.BN_relu(x)
        x = self.Softmax(x)
        x_mask = self.Upsample(x[:,0:1,:,:,:])
        
        x_out = torch.mul(x_in, x_mask)
        
        return x_out, x_mask
    
    
class ResidualBlock(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 conv_num: int):
        super().__init__()
        
        self.Conv_res = ConvBlock(in_channels, out_channels, 1)
        self.Conv = ConvBlock(in_channels, out_channels, 3)        
        
        self.Remain_Conv = nn.ModuleList()
        for i in range(conv_num-1):
            self.Remain_Conv.append(ConvBlock(out_channels, out_channels, 3))
        
    def Residual_forward(self, x_in):
        x_res = self.Conv_res(x_in)
        x = self.Conv(x_in)
        x_out = torch.add(x, x_res)
        
        for Conv in self.Remain_Conv:
            x = Conv(x_out)
            x_out = torch.add(x, x_out)
        return x_out
    
    def forward(self, x_in, x_cross=None, t=None):
        x_out = self.Residual_forward(x_in)
        return x_out


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernels):
        super().__init__()
        
        self.Conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernels, stride=1, padding='same')
        self.BN = nn.BatchNorm3d(out_channels)
        self.ReLU = nn.ReLU()
        
    def forward(self, x_in):
        x = self.Conv(x_in)
        x = self.BN(x)
        x_out = self.ReLU(x)
        
        return x_out

#--------------------------------------------------------------------------------------   
class SwinTransStageBlock(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_layers: int,
                 num_heads: int,
                 window_size: list,
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = True,
                 drop: float = 0.0,
                 attn_drop: float = 0.0,
                 cross_atten: bool = False,
                 is_acmix: bool = True,):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.window_size = window_size
        self.cross_atten = cross_atten
        self.shift_size = tuple(i // 2 for i in window_size)
        self.no_shift = tuple(0 for i in window_size)
        
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            block = SwinTransBlock(embed_dim=embed_dim,
                                   num_heads=num_heads,
                                   window_size=self.window_size,
                                   shift_size=self.no_shift if (i % 2 == 0) else self.shift_size,
                                   mlp_ratio=mlp_ratio,
                                   qkv_bias=qkv_bias,
                                   drop=drop,
                                   attn_drop=attn_drop,
                                   cross_atten=cross_atten,
                                   is_acmix=is_acmix)
            self.blocks.append(block)
        
    def forward(self, x_in, x_cross=None):
        b, c, d, h, w = x_in.shape
        window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
        dp = int(np.ceil(d / window_size[0])) * window_size[0]
        hp = int(np.ceil(h / window_size[1])) * window_size[1]
        wp = int(np.ceil(w / window_size[2])) * window_size[2]
        attn_mask = compute_mask([dp, hp, wp], window_size, shift_size, x_in.device)
        
        x = einops.rearrange(x_in, 'b c d h w -> b d h w c')
        if self.cross_atten:
            x_cross = einops.rearrange(x_cross, 'b c d h w -> b d h w c')
            for block in self.blocks:
                x = block(x, x_cross, attn_mask)
        else:
            for block in self.blocks:
                x = block(x, mask_matrix=attn_mask)
            
        x = x.view(b, d, h, w, -1)
        x_out = einops.rearrange(x, 'b d h w c -> b c d h w')

        return x_out
    

class SwinTransBlock(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 window_size: list,
                 shift_size: list,
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = True,
                 drop: float = 0.0,
                 attn_drop: float = 0.0,
                 cross_atten: bool = False,
                 is_acmix: bool = True):
        super().__init__()
                         
        self.window_size = window_size
        self.shift_size = shift_size
        self.cross_atten = cross_atten
                         
        self.norm1 = nn.LayerNorm(embed_dim)  
        if cross_atten:
            self.attn = MCA_Block(embed_dim,
                                  window_size=window_size,
                                  num_heads=num_heads,
                                  qkv_bias=qkv_bias,
                                  attn_drop=attn_drop,
                                  proj_drop=drop,
                                  is_acmix=is_acmix)
        else:
            self.attn = MSA_Block(embed_dim,
                                  window_size=window_size,
                                  num_heads=num_heads,
                                  qkv_bias=qkv_bias,
                                  attn_drop=attn_drop,
                                  proj_drop=drop,
                                  is_acmix=is_acmix)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP_Block(hidden_size=embed_dim, 
                             mlp_dim=int(embed_dim * mlp_ratio), 
                             dropout_rate=drop)

    def forward_part1(self, x_in, x_cross, mask_matrix):
        x = self.norm1(x_in)
        b, d, h, w, c = x.shape
        window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - d % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - h % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - w % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, dp, hp, wp, _ = x.shape
        dims = [b, dp, hp, wp]
        
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None  
         
        if self.cross_atten:
            x_cross = F.pad(x_cross, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))  
            if any(i > 0 for i in shift_size):
                shifted_x_cross = torch.roll(x_cross, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            else:
                shifted_x_cross = x_cross
            # x_cross_windows = window_partition(shifted_x_cross, window_size)
            # x_windows = window_partition(shifted_x, window_size)
            # attn_windows = self.attn(x_windows, x_cross_windows, mask=attn_mask)
            shifted_x = self.attn(shifted_x, shifted_x_cross, dims, window_size, mask=attn_mask)
        else:
            # x_windows = window_partition(shifted_x, window_size)
            # attn_windows = self.attn(x_windows, mask=attn_mask)
            shifted_x = self.attn(shifted_x, dims, window_size, mask=attn_mask)
        
        # attn_windows = attn_windows.view(-1, *(window_size + (c,)))
        # shifted_x = window_reverse(attn_windows, window_size, dims)
        
        if any(i > 0 for i in shift_size):
            x_out = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x_out = shifted_x

        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x_out = x_out[:, :d, :h, :w, :].contiguous()

        return x_out

    def forward_part2(self, x_in):
        x = self.norm2(x_in)
        x_out = self.mlp(x)
        return x_out

    def forward(self, x_in, x_cross=None, mask_matrix=None):
        x = x_in + self.forward_part1(x_in, x_cross, mask_matrix)
        x_out = x + self.forward_part2(x)
        return x_out


class MSA_Block(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 window_size: list,
                 qkv_bias: bool = False,
                 attn_drop: float = 0.0,
                 proj_drop: float = 0.0,
                 is_acmix: bool = True):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scale = head_dim**-0.5
        self.is_acmix = is_acmix
        mesh_args = torch.meshgrid.__kwdefaults__

        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * self.window_size[0] - 1) * 
                                                                     (2 * self.window_size[1] - 1) * 
                                                                     (2 * self.window_size[2] - 1), num_heads))
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        if mesh_args is not None:
            coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij"))
        else:
            coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.Softmax = nn.Softmax(dim=-1)
        
        if is_acmix:
            self.k_cube = 3 ** 3
            self.k_square = 3 ** 2
            """Fully connected layer in Fig.2"""
            self.fc = nn.Conv2d(3 * self.num_heads, self.k_cube, kernel_size=(1, 1), bias=True)
            """Group convolution layer in Fig.3"""
            self.dep_conv = nn.Conv3d(self.k_cube * embed_dim // self.num_heads, embed_dim, kernel_size=(3, 3, 3), bias=True,
                                      groups=embed_dim // self.num_heads, padding=(1, 1, 1))
            """Weights for both paths"""
            self.weight_alpha = nn.Parameter(torch.tensor(0.5))
            self.weight_beta = nn.Parameter(torch.tensor(0.5))

            """Shift initialization for group convolution"""
            kernel = torch.zeros(self.k_cube, 3, 3, 3)
            for i in range(self.k_cube):
                kernel[i, i // self.k_square, i // 3 % 3, i % 3] = 1.
            kernel = kernel.repeat(self.embed_dim, 1, 1, 1, 1)
            self.dep_conv.weight = nn.Parameter(data=kernel, requires_grad=True)
            self.dep_conv.bias.data.fill_(0.)

    def forward(self, x_in, dims, window_size, mask=None):
        b, d, h, w = dims
        
        if self.is_acmix:
            """fully connected layer"""
            qkv = self.qkv(x_in)
            f_all = qkv.reshape(b, 3 * self.num_heads, d * h * w, -1)
            f_conv = self.fc(f_all).permute(0, 3, 1, 2).reshape(b, self.k_cube * x_in.shape[-1] // self.num_heads, d, h, w)
            
            """group conovlution"""
            out_conv = self.dep_conv(f_conv).permute(0, 2, 3, 4, 1)
            
            """partition windows"""
            qkv = window_partition(qkv, window_size)
            b, _, c = qkv.shape
            qkv = qkv.view(-1, *(window_size + (c,)))
            n = window_size[0] * window_size[1] * window_size[2]
            c = c // 3
            qkv = qkv.reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            x_windows = window_partition(x_in, window_size)
            b, n, c = x_windows.shape
            qkv = self.qkv(x_windows).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.clone()[:n, :n].reshape(-1)
        ].reshape(n, n, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        
        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
        attn = self.Softmax(attn)
        attn = self.attn_drop(attn).to(v.dtype)
        
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        attn_windows = self.proj_drop(x)
        
        """merge windows"""
        attn_windows = attn_windows.view(-1, *(window_size + (c,)))
        shifted_x = window_reverse(attn_windows, window_size, dims)

        if self.is_acmix:
            shifted_x = self.weight_alpha * shifted_x + self.weight_beta * out_conv

        return shifted_x
    

class MCA_Block(nn.Module):

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 window_size: list,
                 qkv_bias: bool = False,
                 attn_drop: float = 0.0,
                 proj_drop: float = 0.0,
                 is_acmix: bool = True):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scale = head_dim**-0.5
        self.is_acmix = is_acmix
        mesh_args = torch.meshgrid.__kwdefaults__

        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * self.window_size[0] - 1) * 
                                                                     (2 * self.window_size[1] - 1) * 
                                                                     (2 * self.window_size[2] - 1), num_heads))
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        if mesh_args is not None:
            coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij"))
        else:
            coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        
        self.q = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.kv = nn.Linear(embed_dim, embed_dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.Softmax = nn.Softmax(dim=-1)

        if is_acmix:
            self.k_cube = 3 ** 3
            self.k_square = 3 ** 2
            """Fully connected layer in Fig.2"""
            self.fc = nn.Conv2d(3 * self.num_heads, self.k_cube, kernel_size=(1, 1), bias=True)
            """Group convolution layer in Fig.3"""
            self.dep_conv = nn.Conv3d(self.k_cube * embed_dim // self.num_heads, embed_dim, kernel_size=(3, 3, 3), bias=True,
                                      groups=embed_dim // self.num_heads, padding=(1, 1, 1))
            """Weights for both paths"""
            self.weight_alpha = nn.Parameter(torch.tensor(0.5))
            self.weight_beta = nn.Parameter(torch.tensor(0.5))

            """Shift initialization for group convolution"""
            kernel = torch.zeros(self.k_cube, 3, 3, 3)
            for i in range(self.k_cube):
                kernel[i, i // self.k_square, i // 3 % 3, i % 3] = 1.
            kernel = kernel.repeat(self.embed_dim, 1, 1, 1, 1)
            self.dep_conv.weight = nn.Parameter(data=kernel, requires_grad=True)
            self.dep_conv.bias.data.fill_(0.)

    def forward(self, x_in, x_cross, dims, window_size, mask=None):
        b, d, h, w = dims
        
        if self.is_acmix:
            """fully connected layer"""
            q = self.q(x_in)
            kv = self.kv(x_cross)
            qkv = torch.cat((q, kv), dim=-1)
            f_all = qkv.reshape(b, 3 * self.num_heads, d * h * w, -1)
            f_conv = self.fc(f_all).permute(0, 3, 1, 2).reshape(b, self.k_cube * x_in.shape[-1] // self.num_heads, d, h, w)
            
            """group conovlution"""
            out_conv = self.dep_conv(f_conv).permute(0, 2, 3, 4, 1)
            
            """partition windows"""
            qkv = window_partition(qkv, window_size)
            b, _, c = qkv.shape
            qkv = qkv.view(-1, *(window_size + (c,)))
            n = window_size[0] * window_size[1] * window_size[2]
            c = c // 3
            qkv = qkv.reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            x_windows = window_partition(x_in, window_size)
            x_cross_windows = window_partition(x_cross, window_size)
            b, n, c = x_windows.shape
            q = self.q(x_windows).reshape(b, n, 1, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
            kv = self.kv(x_cross_windows).reshape(b, n, 2, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
            qkv = torch.cat((q, kv), dim=0)
        
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.clone()[:n, :n].reshape(-1)
        ].reshape(n, n, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        
        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
        attn = self.Softmax(attn)
        attn = self.attn_drop(attn).to(v.dtype)
        
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        attn_windows = self.proj_drop(x)
        
        """merge windows"""
        attn_windows = attn_windows.view(-1, *(window_size + (c,)))
        shifted_x = window_reverse(attn_windows, window_size, dims)

        if self.is_acmix:
            shifted_x = self.weight_alpha * shifted_x + self.weight_beta * out_conv

        return shifted_x
    
    
class MLP_Block(nn.Module):
    def __init__(self, hidden_size: int, mlp_dim: int, dropout_rate: float = 0.0):
        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")
        
        self.linear1 = nn.Linear(hidden_size, mlp_dim)
        self.linear2 = nn.Linear(mlp_dim, hidden_size)
        
        self.drop1 = nn.Dropout(dropout_rate)
        self.drop2 = nn.Dropout(dropout_rate)
        
        self.GELU = nn.GELU()

    def forward(self, x_in):
        x = self.linear1(x_in)
        x = self.GELU(x)
        x = self.drop1(x)
        
        x = self.linear2(x)
        x_out = self.drop2(x)
        
        return x_out
    
    
#--------------------------------------------------------------------------------------    
def compute_mask(dims, window_size, shift_size, device):
    cnt = 0
    d, h, w = dims
    img_mask = torch.zeros((1, d, h, w, 1), device=device)
    for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
        for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
            for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None):
                img_mask[:, d, h, w, :] = cnt
                cnt += 1

    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.squeeze(-1)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

    return attn_mask


def window_partition(x_in, window_size):
    b, d, h, w, c = x_in.shape
    x = x_in.view(b,
                  d // window_size[0],
                  window_size[0],
                  h // window_size[1],
                  window_size[1],
                  w // window_size[2],
                  window_size[2],
                  c)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0] * window_size[1] * window_size[2], c)
    return windows


def window_reverse(windows, window_size, dims):
    b, d, h, w = dims
    x = windows.view(b,
                     d // window_size[0],
                     h // window_size[1],
                     w // window_size[2],
                     window_size[0],
                     window_size[1],
                     window_size[2],
                     -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(b, d, h, w, -1)
    return x


def get_window_size(x_size, window_size, shift_size=None):
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)
    
    
def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
        
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b) 
        return tensor



net = Model()
total = sum([param.nelement() for param in net.parameters()])
# 精确地计算：1MB=1024KB=1048576字节
print('Number of parameter: % .4fM' % (total / 1024 / 1024))
