import math

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
import torch.nn.functional as F
from functools import partial

from .build import BACKBONE_REGISTRY
from .backbone import Backbone
from .resnet import ResNet, BasicBlock, init_pretrained_weights, model_urls

from timm.models.vision_transformer import VisionTransformer as BaseVisionTransformer, _cfg
from .irpe import get_rpe_config, build_rpe

class VisionTransformer(BaseVisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        del self.patch_embed
        del self.head
        num_patches = 49
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, kwargs["embed_dim"]))
        
    def forward(self, x):
        B = x.shape[0]

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)[:, 0]
        return x

class LowrankAttn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.context = nn.Parameter(torch.randn(1, 512), requires_grad=True)
        self.layernorm = nn.LayerNorm(512)
        self.attn = nn.MultiheadAttention(512, num_heads=8, batch_first=True)
        
    def forward(self, x):
        B = x.size(0)
        context = self.context.unsqueeze(0).repeat(B, 1, 1)
        x = self.layernorm(x)
        out, _ = self.attn(context, x, x)
        return out

def reshape_to_batch(x, num_heads):
    batch_size, seq_len, in_feature = x.size()
    sub_dim = torch.div(in_feature, num_heads, rounding_mode='trunc')
    return x.reshape(batch_size, seq_len, num_heads, sub_dim)\
            .permute(0, 2, 1, 3)\
            .reshape(batch_size * num_heads, seq_len, sub_dim)

def reshape_from_batch(x, num_heads, is_rel=False):
    if not is_rel:
        batch_size, seq_len, in_feature = x.size()
        batch_size = torch.div(batch_size, num_heads, rounding_mode='trunc')
        x = x.reshape(batch_size, num_heads, seq_len, in_feature)
    else:
        batch_size, _, seq_len, in_feature = x.size()
    
    out_dim = in_feature * num_heads
    
    return x.permute(0, 2, 1, 3)\
            .reshape(batch_size, seq_len, out_dim)


class ResnetFeature(ResNet):        
    def __init__(
        self, img_size=224, cp_layer=4, **kwargs
    ):
        super().__init__(**kwargs)
        
        self.cp_layer = cp_layer
        if self.cp_layer == 4:
            self.out_dim = 512
            self.num_tokens = (img_size // 32)**2
        elif self.cp_layer == 3:
            self.out_dim = 256
            self.num_tokens = (img_size // 16) ** 2
            del self.layer4
        elif self.cp_layer == 2:
            self.out_dim = 128
            self.num_tokens = (img_size // 8) ** 2
            del self.layer3
            del self.layer4
        else:
            raise Exception("Parameter is not correct")  
        
    def featuremaps(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        if "layer1" in self.ms_layers:
            x = self.mixstyle(x)
        x = self.layer2(x)
        
        if self.cp_layer >= 3:
            if "layer2" in self.ms_layers:
                x = self.mixstyle(x)
            x = self.layer3(x)
            
        if self.cp_layer >= 4:            
            if "layer3" in self.ms_layers:
                x = self.mixstyle(x)
            x = self.layer4(x)

        return x

        
    def forward(self, x):
        f = self.featuremaps(x)
        B, d, _, _ = f.shape
        f = f.view(B, d, -1).permute(0, 2, 1)
        return f

class SparseResidual(nn.Module):
    def __init__(self, embed_dim, thresh) -> None:
        super().__init__()
        
        self.mask_embedding = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim))
        # self.thresh = nn.Parameter(torch.tensor(thresh), requires_grad=True)
        self.thresh = thresh
        self.mlp = nn.Sequential(nn.Linear(embed_dim, embed_dim))
        
    def forward(self, tokens, x):
        mask = self.mask_embedding(torch.cat([tokens, x], -1))
        mask = torch.sigmoid(mask)
        
        # thresholding the mask, make it sparse
        if self.thresh > 0.0:
            mask = F.relu(mask - self.thresh)
            
            # recover the mask value
            tmp = (mask > 0).float()
            mask = mask + tmp * self.thresh
        
        # print(mask)
        return tokens * (1 - mask) + mask * self.mlp(x)
        

class InLay(nn.Module):
    def __init__(self, dim=512, num_tokens=49, learnable_dim=512, 
                 has_sparse_res=False, dropout=0.0) -> None:
        super().__init__()
        
        self.linear_q = nn.Sequential(nn.Linear(dim, dim),
                                      nn.GELU(),
                                      nn.Dropout(dropout),
                                      nn.Linear(dim, 4 * dim),
                                      nn.Dropout(dropout))
        
        self.linear_k = nn.Sequential(nn.Linear(dim, dim),
                                      nn.GELU(),
                                      nn.Dropout(dropout),
                                      nn.Linear(dim, 4 * dim),
                                      nn.Dropout(dropout))
        
        self.tokens = nn.Parameter(torch.randn(num_tokens, learnable_dim),
                                   requires_grad=True)
        
        
        self.proj = nn.Linear(learnable_dim, dim)
        
        self.has_sparse_res = has_sparse_res
        
        if self.has_sparse_res:
            self.sparse_residual = SparseResidual(dim, 0.5)
        
        self.num_heads = 32
        
    def process_tokens(self, x):
        B = x.size(0)
        tokens = self.tokens.unsqueeze(0).repeat(B, 1, 1)
        # tokens = tokens + self.lowrankattn(x)

        return tokens
    
    def apply_adj(self, adj, tokens):
        value = reshape_to_batch(tokens, self.num_heads)
        out = adj.matmul(value)
        out = reshape_from_batch(out, self.num_heads) 
        
        return out

    def compute_adjacency(self, x):
        query, key = self.linear_q(x), self.linear_k(x)
        query = reshape_to_batch(query, self.num_heads)
        key = reshape_to_batch(key, self.num_heads)
        d = query.size(-1)

        
        dot_prod = query.matmul(key.transpose(-2, -1)) / math.sqrt(d)
        
        adjacency_matrix = torch.tanh(dot_prod)
        return adjacency_matrix
    
    def inLay(self, x, tokens):
        adjacency_matrix = self.compute_adjacency(x)
        out = self.apply_adj(adjacency_matrix, tokens)
                
        out = self.proj(out)
        return out

    def forward(self, x):        
        tokens = self.process_tokens(x)
        
        out = self.inLay(x, tokens)
        
        if self.has_sparse_res:
            out = self.sparse_residual(out, x)
        
        return out

class InLayWithMem(InLay):
    def __init__(self, dim=512, num_tokens=49, learnable_dim=512, 
                 has_sparse_res=False, dropout=0.0) -> None:
        super().__init__(dim=dim, num_tokens=num_tokens, learnable_dim=learnable_dim,
                         has_sparse_res=has_sparse_res, dropout=dropout)
                        
        self.memory_reader = nn.MultiheadAttention(512, num_heads=8, kdim=512, vdim=512,
                                                   batch_first=True)
        
        self.memory = nn.Parameter(torch.randn(32, 512),
                                   requires_grad=True)
        
        
    
    def forward(self, x):
        B = x.size(0)
        memory = self.memory.unsqueeze(0).repeat(B, 1, 1)

        internal = super().forward(x)
        prototype, _ = self.memory_reader(x, memory, memory)
        
        return internal + prototype, (prototype, x.detach())
        

class AttentionWithQuantize(InLay):
    def __init__(self, dim=512, num_tokens=49, learnable_dim=512, 
                 has_sparse_res=False, dropout=0.0) -> None:
        super().__init__(dim=dim, num_tokens=num_tokens, learnable_dim=learnable_dim,
                         has_sparse_res=has_sparse_res, dropout=dropout)
        self.quant = Quantization(n=16, dim=512)
    
    def forward(self, x):
        internal = super().forward(x)
        codebook, additional_out = self.quant(x)
        
        return internal + codebook, additional_out

# class AttentionWithQuantize(InLay):
#     def __init__(self, dim=512, num_tokens=49, learnable_dim=512, 
#                  has_sparse_res=False, dropout=0.0) -> None:
#         super().__init__(dim=dim, num_tokens=num_tokens, learnable_dim=learnable_dim,
#                          has_sparse_res=has_sparse_res, dropout=dropout)
#         self.quant = Quantization(n=16, dim=512)
#         del self.tokens
    
#     def process_tokens(self, x):
#         tokens, additional_out = self.quant(x)

#         return tokens, additional_out
    
    
#     def forward(self, x):
#         tokens, additional_out = self.process_tokens(x)
        
#         out = self.inLay(x, tokens)
#         return out, additional_out

class Quantization(nn.Module):
    def __init__(self, n=128, dim=512):
        super().__init__()
        self.n = n
        self.code_book = nn.Parameter(torch.randn(self.n, dim),
                                      requires_grad=True)    
    
        self.enc = nn.Linear(dim, dim)
    
    def forward(self, x):
        original_shape = x.size()
        x = x.view(-1, x.size(-1))
        B = x.size(0)
        z = self.enc(x)
        
        code_book = self.code_book.unsqueeze(0).repeat(B, 1, 1)
        distance = torch.norm(z.unsqueeze(1) - code_book, dim=-1)
        _, argmin = distance.min(-1)
        
        mask = torch.zeros(B, self.n, device=x.device).scatter_(1, argmin.unsqueeze(1), 1).unsqueeze(-1).bool()
        q = torch.masked_select(code_book, mask).reshape(B, -1)
        
        q = z + (q - z).detach()
        
        return q.reshape(original_shape), (q, z)



class SelectiveInLay(InLay):
    def __init__(self, dim=512, num_tokens=49, learnable_dim=512, 
                 has_sparse_res=False, dropout=0.0) -> None:
        super().__init__(dim=dim, num_tokens=num_tokens, learnable_dim=learnable_dim,
                         has_sparse_res=has_sparse_res, dropout=dropout)

        del self.tokens
         
        self.ignore_token = nn.Parameter(torch.randn(1, 512),
                                   requires_grad=True)
        
        self.num_tokens = 16
        self.max_tokens = 64
        self.tokens = nn.Parameter(torch.randn(self.max_tokens, 512),
                                   requires_grad=True)
        
        self.proj = nn.Linear(512, 512)
        
        self.token_selective_attn = nn.MultiheadAttention(512, 8, batch_first=True)
        self.layernorm = nn.LayerNorm(512)
        self.act = nn.GELU()
    
    def forward(self, x):
        B, N, d = x.size()
        updated_x = torch.cat([x, self.ignore_token.expand(B, 1, d)], 1)
        
        tokens = self.tokens.unsqueeze(0).repeat(B, 1, 1)
        updated_x, weight = self.token_selective_attn(tokens, updated_x, updated_x)
        
        _, indices = torch.topk(weight[:, -1], largest=False, k=self.num_tokens)
        mask = torch.zeros(B, self.max_tokens, device=x.device).scatter_(1, indices, 1).unsqueeze(-1).bool()
        
        tokens = torch.masked_select(tokens, mask).reshape(B, -1, d)
        x = torch.masked_select(updated_x, mask).reshape(B, -1, d)
        
        out = self.inLay(self.layernorm(x), tokens)
        
        if self.has_sparse_res:
            out = self.sparse_residual(out, x)
        
        return out

class InLayFuse(InLay):
    def __init__(self, dim=512, num_tokens=49, learnable_dim=512, 
                 has_sparse_res=False, dropout=0.0) -> None:
        super().__init__(dim=dim, num_tokens=num_tokens, learnable_dim=learnable_dim,
                         has_sparse_res=has_sparse_res, dropout=dropout)

        # self.linear_v = nn.Linear(512, 512)
        # self.proj_v = nn.Linear(512, 512)

        self.sparse_residual = SparseResidual(512, 0.0)
        
    # def inLay(self, x, tokens):
    #     adjacency_matrix = self.compute_adjacency(x)
        
    #     tokens = self.proj(self.apply_adj(adjacency_matrix, tokens))
    #     v = self.proj_v(self.apply_adj(adjacency_matrix, self.linear_v(x)))
        
    #     return tokens, v
    
    # def forward(self, x):
    #     tokens = self.process_tokens(x)
    #     tokens, v = self.inLay(x, tokens)
        
    #     out = self.sparse_residual(tokens, v.detach())
            
    #     return out, (tokens.detach(), out)

    def forward(self, x):        
        tokens = self.process_tokens(x)
        
        tokens = self.inLay(x, tokens)
        
        out = self.sparse_residual(tokens, x.detach())
        
        return out, (tokens.detach(), out)


class InLayRel(InLay):
    def __init__(self, dim=512, num_tokens=49, learnable_dim=512, 
                 has_sparse_res=False, dropout=0.0) -> None:
        super().__init__(dim=dim, num_tokens=num_tokens, learnable_dim=learnable_dim,
                         has_sparse_res=has_sparse_res, dropout=dropout)

        self.tokens = None
        rpe_config = get_rpe_config(
            ratio=1.9,
            method='product',
            mode='ctx',
            shared_head=True,
            skip=0,
            rpe_on='v',
        )
        _, _, self.rpe_v = build_rpe(rpe_config,
                    head_dim=512//32,
                    num_heads=32)

    def process_tokens(self, x):
        return None

    def apply_adj(self, adj, tokens):
        BH, N, M = adj.shape
        out = self.rpe_v(adj.view(-1, self.num_heads, N, M))
        out = reshape_from_batch(out, self.num_heads, is_rel=True)
        
        return out

    
class AVGPool2D(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, ind):
        B, N, D = ind.size()
        H = int(math.sqrt(N))
        out = self.avgpool(ind.permute(0, 2, 1).view(B, D, H, H))
        return out.view(out.size(0), -1)
    
class AVGPool1D(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, ind):
        B, N, D = ind.size()
        return ind.mean(1)
    
    
class CustomAttention(Backbone):
    def __init__(self, img_size=224, resnet_out_layer=4, learnable_dim=512, type="inLay", has_sparse_res=False) -> None:
        super().__init__()
        self.resnet = ResnetFeature(img_size=img_size, cp_layer=resnet_out_layer, block=BasicBlock, layers=[2, 2, 2, 2])
        dim = self.resnet.out_dim
        num_tokens = self.resnet.num_tokens
        
        init_pretrained_weights(self.resnet, model_urls["resnet18"])

        self._out_features = dim
        self.layernorm = nn.LayerNorm(dim)
        
        if type == "inlay":
            self.attn = InLay(dim=dim, num_tokens=num_tokens, 
                              has_sparse_res=has_sparse_res, learnable_dim=learnable_dim)
        elif type == "inlay_rel":
            self.attn = InLayRel(dim=dim, num_tokens=num_tokens, learnable_dim=learnable_dim, 
                                 has_sparse_res=has_sparse_res)
        elif type == "inlay_selective":
            self.attn = SelectiveInLay(dim=dim, num_tokens=num_tokens, learnable_dim=learnable_dim, 
                                       has_sparse_res=has_sparse_res)
        elif type == "inlay_fuse":
            self.attn = InLayFuse(dim=dim, num_tokens=num_tokens, 
                              has_sparse_res=has_sparse_res, learnable_dim=learnable_dim)
        elif type == "inlay_mem":
            self.attn = InLayWithMem(dim=dim, num_tokens=num_tokens,
                                     has_sparse_res=has_sparse_res, learnable_dim=learnable_dim)
        elif type == "inlay_quant":
            self.attn = AttentionWithQuantize(dim=dim, num_tokens=num_tokens, 
                                              has_sparse_res=has_sparse_res, learnable_dim=learnable_dim)
        elif type == "attention":
            self.attn = Attention()
        elif type == "attentioninlay":
            self.attn = AttentionInLay()

        self.avgpool = AVGPool2D()
        if type == "inlay_selective":
            self.avgpool = AVGPool1D()

    def forward(self, x):
        # return self.vit(self.ind(x))
        out = self.attn(self.layernorm(self.resnet(x)))
        
        if type(out) == tuple:
            out, additional_out = out
            return self.avgpool(out), additional_out
        else:
            return self.avgpool(out)
        
        
    def get_thresh(self):
        return self.attn.get_thresh()
            
class Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(512, num_heads=8, batch_first=True)
        
    def forward(self, x):
        x = self.attn(x, x, x)[0]
        return x
        
class AttentionInLay(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.tokens = nn.Parameter(torch.randn(49, 512),
                                   requires_grad=True)
        self.attn_internal = nn.MultiheadAttention(embed_dim=512, num_heads=16, batch_first=True)
        self.attn_external = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
        # self.lowrank_attn = LowrankAttn()
        self.layernorm = nn.LayerNorm(512)
        self.proj = nn.Identity()

    def forward(self, x):
        print(x.shape)
        B = x.size(0)
        tokens = self.tokens.unsqueeze(0).repeat(B, 1, 1)
        internal, _ = self.attn_internal(x, x, tokens)
        attended, _ = self.attn_external(self.layernorm(internal), x, x)
        
        return self.proj(attended + internal)
        

@BACKBONE_REGISTRY.register()
def ind(cfg, **kwargs):
    model = CustomAttention(img_size=cfg.ATTN.IMG_SIZE,
                            resnet_out_layer=cfg.ATTN.RESNET_OUT_LAYER,
                            learnable_dim=cfg.ATTN.LEARNABLE_DIM, 
                            type=cfg.ATTN.TYPE, 
                            has_sparse_res=cfg.ATTN.SPARSE_RES)
    return model
