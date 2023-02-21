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
        
        print(mask)
        return tokens * (1 - mask) + mask * self.mlp(x)
        



class InLay(nn.Module):
    def __init__(self, has_sparse_res=False, dropout=0.1) -> None:
        super().__init__()
        
        self.linear_q = nn.Sequential(nn.Linear(512, 512),
                                      nn.GELU(),
                                      nn.Dropout(dropout),
                                      nn.Linear(512, 2048),
                                      nn.Dropout(dropout))
        
        self.linear_k = nn.Sequential(nn.Linear(512, 512),
                                      nn.GELU(),
                                      nn.Dropout(dropout),
                                      nn.Linear(512, 2048),
                                      nn.Dropout(dropout))
        
        self.tokens = nn.Parameter(torch.randn(49, 512),
                                   requires_grad=True)
        
        
        self.proj = nn.Linear(512, 512)
        
        self.has_sparse_res = has_sparse_res
        
        if self.has_sparse_res:
            self.sparse_residual = SparseResidual(512, 0.5)
        
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

class SelectiveInLay(InLay):
    def __init__(self, has_sparse_res=False, dropout=0) -> None:
        super().__init__(has_sparse_res, dropout)       
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
    def __init__(self, has_sparse_res=False, dropout=0.1) -> None:
        super().__init__(has_sparse_res, dropout)
        self.linear_v = nn.Linear(512, 512)
        self.proj_v = nn.Linear(512, 512)

        if self.has_sparse_res:
            self.sparse_residual = SparseResidual(512, 0.0)
        
    def inLay(self, x, tokens):
        adjacency_matrix = self.compute_adjacency(x)
        v = reshape_to_batch(self.linear_v(x), self.num_heads)
        
        tokens = self.proj(self.apply_adj(adjacency_matrix, tokens))
        v = self.proj_v(self.apply_adj(adjacency_matrix, v))
        
        return tokens, v
    
    def forward(self, x):
        tokens = self.process_tokens(x)
        tokens, v = self.inLay(x, tokens)
        
        tokens = self.sparse_residual(tokens, v)
            
        return tokens, (tokens, )

         

        
        
        
        
        


class InLayRel(InLay):
    def __init__(self, has_sparse_res=False, dropout=0.0) -> None:
        super().__init__(has_sparse_res, dropout)
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
    def __init__(self, type="inLay", has_sparse_res=False) -> None:
        super().__init__()
        self.resnet = ResnetFeature(block=BasicBlock, layers=[2, 2, 2, 2])
        init_pretrained_weights(self.resnet, model_urls["resnet18"])

        self._out_features = 512
        self.layernorm = nn.LayerNorm(512)
        
        if type == "inlay":
            self.attn = InLay(has_sparse_res=has_sparse_res)
        elif type == "inlay_rel":
            self.attn = InLayRel(has_sparse_res=has_sparse_res)
        elif type == "inlay_selective":
            self.attn = SelectiveInLay(has_sparse_res=has_sparse_res)
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
            out, tokens = out
            return self.avgpool(out), tokens
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
        self.attn = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
        # self.lowrank_attn = LowrankAttn()
        self.sparse_residual = SparseResidual(512, 0.0)
        self.proj = nn.Identity()

    def forward(self, x):
        B = x.size(0)
        tokens = self.tokens.unsqueeze(0).repeat(B, 1, 1)
        out1, weight = self.attn(x, x, x)
        out2 = weight.matmul(tokens)
        x = self.sparse_residual(out2, out1)
        
        return self.proj(x), (out2, x)
        

@BACKBONE_REGISTRY.register()
def ind(cfg, **kwargs):
    model = CustomAttention(type=cfg.ATTN.TYPE, has_sparse_res=cfg.ATTN.SPARSE_RES)
    return model
