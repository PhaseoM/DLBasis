import torch
from . import attlib
from torch import nn
from torchvision.ops import StochasticDepth


class AttentionBlock(nn.Module):
    def __init__(
        self,
        num_heads=8,
        d_model=512,
        mlp_hiddens=2048,
        dropout=0,
        drop_path_rate=0,
        att_score=None,
    ):
        super().__init__()
        if att_score is None:
            att_score = attlib.ScaledDotAttention()
        self.att_model = nn.Sequential(
            nn.LayerNorm(normalized_shape=d_model),
            attlib.MultiHeadAttention(
                num_heads=num_heads,
                d_model=d_model,
                att_score=att_score,
            ),
        )
        self.mlp_model = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(in_features=d_model, out_features=mlp_hiddens),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=mlp_hiddens, out_features=d_model),
            nn.Dropout(dropout),
        )
        self.att_drop_path = StochasticDepth(p=drop_path_rate, mode="row")
        self.mlp_drop_path = StochasticDepth(p=drop_path_rate, mode="row")

    def forward(self, X):
        X = X + self.att_drop_path(self.att_model(X))
        X = X + self.mlp_drop_path(self.mlp_model(X))
        return X


def _att_seqs_stack(size_n, num_heads, d_model, mlp_hiddens, dropout, drop_path_rate):
    drop_path_rates = torch.linspace(
        0,
        drop_path_rate,
        size_n,
    ).tolist()
    att_seqs = []
    for index_layers in range(size_n):
        att_seqs.append(
            AttentionBlock(
                num_heads=num_heads,
                d_model=d_model,
                mlp_hiddens=mlp_hiddens,
                dropout=dropout,
                drop_path_rate=drop_path_rates[index_layers],
            )
        )
    return att_seqs


class SequentialTranspose(nn.Module):
    def __init__(self, dim0, dim1):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, X):
        return X.transpose(self.dim0, self.dim1)


class ImgPatchEmbedding(nn.Module):
    def __init__(
        self,
        img_size,
        patch_size,
        d_model,
        dropout,
    ):
        super().__init__()

        def _make_tuple(x):
            if not isinstance(x, (list, tuple)):
                return (x, x)
            return x

        img_size, patch_size = _make_tuple(img_size), _make_tuple(patch_size)
        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]) + 1
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02, a=-0.04, b=0.04)
        self.patch_embed = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=d_model,
                kernel_size=patch_size,
                stride=patch_size,
            ),
            nn.Flatten(2),
            SequentialTranspose(1, 2),
        )
        self.pos_embed = nn.Parameter(torch.empty(1, self.num_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02, a=-0.04, b=0.04)
        self.dropout_embed = nn.Dropout(dropout)

    def forward(self, X):
        cls_token_ = self.cls_token.expand(X.shape[0], -1, -1)
        patch_embed_ = self.patch_embed(X)
        patch_embed_cls = torch.cat((cls_token_, patch_embed_), dim=1)
        pos_embed_ = self.pos_embed.expand(X.shape[0], -1, -1)
        output = self.dropout_embed(patch_embed_cls + pos_embed_)
        return output


class VisionTransformer(nn.Module):
    def __init__(
        self,
        size_n=12,
        d_model=768,
        num_heads=12,
        num_classes=10,
        mlp_hiddens=3072,
        img_size=32,
        patch_size=4,
        dropout=0,
        drop_path_rate=0.1,
    ):
        super().__init__()
        self.embedding = ImgPatchEmbedding(img_size, patch_size, d_model, dropout)
        self.encoder = nn.Sequential(
            *_att_seqs_stack(size_n, num_heads, d_model, mlp_hiddens, dropout, drop_path_rate),
        )
        self.mlp_cls = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(in_features=d_model, out_features=num_classes),
        )

    def forward(self, X):
        X = self.embedding(X)
        cls_token = self.encoder(X)[:, 0]
        output = self.mlp_cls(cls_token)
        return output
