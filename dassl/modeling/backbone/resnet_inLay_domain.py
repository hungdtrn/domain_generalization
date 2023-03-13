import torch
import torch.nn as nn

from .resnet_inLay import InLay as BaseInLay, ResnetFeature, AVGPool1D, AVGPool2D, Backbone, BACKBONE_REGISTRY, init_pretrained_weights, BasicBlock, model_urls

class InLay(BaseInLay):
    def __init__(self, dim=512, num_tokens=49, learnable_dim=512, 
                 residual_type=0, dropout=0.0) -> None:
        super().__init__(dim=dim, num_tokens=num_tokens, learnable_dim=learnable_dim,
                         residual_type=residual_type, dropout=dropout)


        # self.linear_q = nn.Sequential(nn.Linear(dim, dim//4),
        #                               nn.GELU(),
        #                               nn.Dropout(dropout),
        #                               nn.Linear(dim//4, dim),
        #                               nn.Dropout(dropout))
        
        # self.linear_k = nn.Sequential(nn.Linear(dim, dim//4),
        #                               nn.GELU(),
        #                               nn.Dropout(dropout),
        #                               nn.Linear(dim//4, dim),
        #                               nn.Dropout(dropout))
        self.linear_q = nn.Sequential(nn.Linear(dim, dim),
                                      nn.LayerNorm(dim),
                                      nn.Linear(dim, dim))
        
        self.linear_k = nn.Sequential(nn.Linear(dim, dim),
                                      nn.LayerNorm(dim),
                                      nn.Linear(dim, dim),)


        self.num_heads = 8

        
    def forward(self, x):
        
        return super().forward(x)        

class DomainSpecificAttn(Backbone):
    def __init__(self, img_size=224, resnet_out_layer=4, learnable_dim=512, type="inLay", residual_type=0) -> None:
        super().__init__()
        self.resnet = ResnetFeature(img_size=img_size, cp_layer=resnet_out_layer, block=BasicBlock, layers=[2, 2, 2, 2])
        dim = self.resnet.out_dim
        num_tokens = self.resnet.num_tokens
        
        init_pretrained_weights(self.resnet, model_urls["resnet18"])

        self._out_features = dim
        self.layernorm = nn.LayerNorm(dim)
        
        self.dim = dim
        self.domain_embedding = nn.Linear(5, dim * 32 * 2)

        
        attn_class = None
        if type == "inlay":
            attn_class = InLay
        elif type == "attention":
            self.attn = Attention(dim=dim)

        if type not in ["attention", "attentioninlay"]:
            self.attn = attn_class(dim=dim, num_tokens=num_tokens, 
                              residual_type=residual_type, learnable_dim=learnable_dim)

        self.avgpool = AVGPool2D()
        if type == "inlay_selective":
            self.avgpool = AVGPool1D()

    def domain_specific_embedding(self, x, domain):
        if domain is not None:
            domain = nn.functional.one_hot(domain, num_classes=4)
        else:
            domain = torch.zeros(len(x), 4).to(x.device)
            
        shared = torch.tensor([1]).expand(len(x), 1).to(domain.device)
        
        w_d = self.domain_embedding(torch.cat([domain, shared], -1).float())
        w_d1, w_d2 = torch.chunk(w_d, 2, dim=-1)
        w_d1 = w_d1.view(len(x), self.dim, -1)
        w_d2 = w_d2.view(len(x), -1, self.dim)
        
        return ((x @ w_d1) @ w_d2)

    def forward(self, input):
        x, domain = input
        
        # return self.vit(self.ind(x))
        feat = self.layernorm(self.resnet(x))
        feat = self.domain_specific_embedding(feat, domain)

        out = self.attn(feat)
        
        if type(out) == tuple:
            out, additional_out = out
            return self.avgpool(out), additional_out
        else:
            return self.avgpool(out)
        
        
    def get_thresh(self):
        return self.attn.get_thresh()
            
class Attention(nn.Module):
    def __init__(self, dim=512) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads=8, batch_first=True)
        
    def forward(self, x):
        x = self.attn(x, x, x)[0]
        return x
    
@BACKBONE_REGISTRY.register()
def ind_domain(cfg, **kwargs):
    model = DomainSpecificAttn(img_size=cfg.ATTN.IMG_SIZE,
                            resnet_out_layer=cfg.ATTN.RESNET_OUT_LAYER,
                            learnable_dim=cfg.ATTN.LEARNABLE_DIM, 
                            type=cfg.ATTN.TYPE, 
                            residual_type=cfg.ATTN.RESIDUAL_TYPE)
    return model
