import torch
from torch import Tensor
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from typing import List, Tuple, Type
import math
from segment_anything.modeling.common import MLPBlock


class LayerNorm3d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
        return x

class Adapter(nn.Module):
    def __init__(self, input_dim, mid_dim):
        super().__init__()
        self.model = MLP(
            input_dim=input_dim, hidden_dim=mid_dim, output_dim=input_dim, num_layers=2
        )

    def forward(self, features):
        out = features + self.model(features)
        return out


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


class TwoWayTransformer(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.
        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
        """
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )
            )

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_coord,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
          image_embedding (torch.Tensor): image to attend to. Should be shape
            B x embedding_dim x h x w for any h and w.
          image_pe (torch.Tensor): the positional encoding to add to the image. Must
            have the same shape as image_embedding.
          point_embedding (torch.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x embedding_dim for any N_points.
        Returns:
          torch.Tensor: the processed point_embedding
          torch.Tensor: the processed image_embedding
        """
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        point_embedding = F.grid_sample(image_embedding, point_coord, align_corners=False).squeeze(2).squeeze(2)
        point_pe = F.grid_sample(image_pe, point_coord, align_corners=False).squeeze(2).squeeze(2)

        point_pe = point_pe.permute(0, 2, 1)
        point_embedding = point_embedding.permute(0, 2, 1)
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)

        image_pe = image_pe.flatten(2).permute(0, 2, 1)
        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            image_embedding, point_embedding = layer(
                image_embedding,
                point_embedding,
                image_pe,
                point_pe,
            )
        return image_embedding

class TwoWayAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.
        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
        """
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.global_query = nn.parameter.Parameter(data=0.1 * torch.randn(1, 10, embedding_dim))

    def forward(self, img_embed, point_embed, img_pe, point_pe) -> Tuple[Tensor, Tensor]:
        print("ww:", img_embed.shape, point_embed.shape)
        q = torch.cat([self.global_query, point_embed], dim=1)
        self_out = self.self_attn(q=q, k=q, v=q)
        self_out = self.norm1(self_out)

        # Cross attention block, tokens attending to image embedding
        queries = q + self_out
        queries = self.norm2(queries)
        point_embed = queries[:, 10:, :]
        queries = queries[:, :10, :]

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        attn_out = self.cross_attn_image_to_token(q=img_embed, k=queries, v=queries)
        
        keys = img_embed + attn_out
        keys = self.norm4(keys)

        return keys, point_embed


class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)
        return out

class Attention_Malignancy(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."
        self.pos_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.neg_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v_pos: Tensor, v_neg: Tensor) -> Tensor:
        # Input projections
        k = self.k_proj(k)
        q = self.q_proj(q)
        k = self._separate_heads(k, self.num_heads)
        q = self._separate_heads(q, self.num_heads)
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)
        # Separate into heads
        if v_pos != None:
            v_pos = self.pos_proj(v_pos)
            v_pos = self._separate_heads(v_pos, self.num_heads)            
            # Get output
            out_pos = attn @ v_pos
            out_pos = self._recombine_heads(out_pos)
            out_pos = self.out_proj(out_pos)
        else:
            out_pos = None
        if v_neg != None:
            v_neg = self.neg_proj(v_neg)
            v_neg = self._separate_heads(v_neg, self.num_heads)
            out_neg = attn @ v_neg
            out_neg = self._recombine_heads(out_neg)
            out_neg = self.out_proj(out_neg)
        else:
            out_neg = None
        return out_pos, out_neg

class Attention_Subtype(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."
        self.proj_0 = nn.Linear(embedding_dim, self.internal_dim)
        self.proj_1 = nn.Linear(embedding_dim, self.internal_dim)
        self.proj_2 = nn.Linear(embedding_dim, self.internal_dim)
        self.proj_3 = nn.Linear(embedding_dim, self.internal_dim)
        self.proj_4 = nn.Linear(embedding_dim, self.internal_dim)
        self.proj_5 = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor, i) -> Tensor:
        # Input projections
        k = self.k_proj(k)
        q = self.q_proj(q)
        k = self._separate_heads(k, self.num_heads)
        q = self._separate_heads(q, self.num_heads)
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Separate into heads
        counter = [self.proj_0, self.proj_1, self.proj_2, self.proj_3, self.proj_4, self.proj_5]
        v = counter[i](v)
        v = self._separate_heads(v, self.num_heads)            
        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out

class Attention_Stage(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."
        self.proj_0 = nn.Linear(embedding_dim, self.internal_dim)
        self.proj_1 = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor, i) -> Tensor:
        # Input projections
        k = self.k_proj(k)
        q = self.q_proj(q)
        k = self._separate_heads(k, self.num_heads)
        q = self._separate_heads(q, self.num_heads)
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Separate into heads
        counter = [self.proj_0, self.proj_1]
        v = counter[i](v)
        v = self._separate_heads(v, self.num_heads)            
        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out

class Attention_Grade(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."
        self.proj_0 = nn.Linear(embedding_dim, self.internal_dim)
        self.proj_1 = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor, i) -> Tensor:
        # Input projections
        k = self.k_proj(k)
        q = self.q_proj(q)
        k = self._separate_heads(k, self.num_heads)
        q = self._separate_heads(q, self.num_heads)
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Separate into heads
        counter = [self.proj_0, self.proj_1]
        v = counter[i](v)
        v = self._separate_heads(v, self.num_heads)            
        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out

class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))

class TaskPromptEncoder_Subtype(nn.Module):
    def __init__(
            self,
            embedding_dim=256,
            mlp_dim=2048,
            mla_channels=256, 
            mlahead_channels=128, 
            num_heads=8
    ):
        super().__init__()
        self.cls_embedding_0 = nn.parameter.Parameter(torch.randn(1, 4096, 256))
        self.cls_embedding_1 = nn.parameter.Parameter(torch.randn(1, 4096, 256))
        self.cls_embedding_2 = nn.parameter.Parameter(torch.randn(1, 4096, 256))
        self.cls_embedding_3 = nn.parameter.Parameter(torch.randn(1, 4096, 256))
        self.cls_embedding_4 = nn.parameter.Parameter(torch.randn(1, 4096, 256))
        self.cls_embedding_5 = nn.parameter.Parameter(torch.randn(1, 4096, 256))
        self.attn = Attention_Subtype(embedding_dim, num_heads)
        self.mlp = MLPBlock(embedding_dim, mlp_dim, nn.ReLU)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.head = nn.Sequential(nn.Conv3d(mla_channels, mlahead_channels, 3, padding=1, bias=False),
                     nn.InstanceNorm3d(mlahead_channels),
                     nn.ReLU(),
                     nn.Conv3d(mlahead_channels, mlahead_channels, 3, padding=1, bias=False),
                     nn.InstanceNorm3d(mlahead_channels),
                     nn.ReLU())
        
    def forward(self, x, i):
        x = F.interpolate(x, scale_factor = 0.5, mode='trilinear', align_corners=True)
        x = x.flatten(2).permute(0, 2, 1)
        candidates = [self.cls_embedding_0, self.cls_embedding_1, self.cls_embedding_2, self.cls_embedding_3, self.cls_embedding_4, self.cls_embedding_5]
        cls_embedding = candidates[i] 
        x_pos = self.attn(q=x, k=x, v=cls_embedding, i=i)
        x_pos = x + x_pos
        x_pos = x_pos + self.mlp(self.norm1(x_pos))
        x_pos = self.norm2(x_pos)
        x_pos = x_pos.transpose(1,2).reshape([1, -1, 16, 16, 16])
        x_pos = F.interpolate(x_pos, scale_factor = 2, mode='trilinear', align_corners=True)
        head_pos = F.interpolate(self.head(x_pos), scale_factor = 2, mode='trilinear', align_corners=True)
        return head_pos

class TaskPromptEncoder_Stage(nn.Module):
    def __init__(
            self,
            embedding_dim=256,
            mlp_dim=2048,
            mla_channels=256, 
            mlahead_channels=128, 
            num_heads=8
    ):
        super().__init__()
        self.cls_embedding_0 = nn.parameter.Parameter(torch.randn(1, 4096, 256))
        self.cls_embedding_1 = nn.parameter.Parameter(torch.randn(1, 4096, 256))
        self.attn = Attention_Stage(embedding_dim, num_heads)
        self.mlp = MLPBlock(embedding_dim, mlp_dim, nn.ReLU)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.head = nn.Sequential(nn.Conv3d(mla_channels, mlahead_channels, 3, padding=1, bias=False),
                     nn.InstanceNorm3d(mlahead_channels),
                     nn.ReLU(),
                     nn.Conv3d(mlahead_channels, mlahead_channels, 3, padding=1, bias=False),
                     nn.InstanceNorm3d(mlahead_channels),
                     nn.ReLU())
        
    def forward(self, x, i):
        x = F.interpolate(x, scale_factor = 0.5, mode='trilinear', align_corners=True)
        x = x.flatten(2).permute(0, 2, 1)
        candidates = [self.cls_embedding_0, self.cls_embedding_1]
        
        cls_embedding = candidates[i] 
        x_pos = self.attn(q=x, k=x, v=cls_embedding, i=i)
        x_pos = x + x_pos
        x_pos = x_pos + self.mlp(self.norm1(x_pos))
        x_pos = self.norm2(x_pos)
        x_pos = x_pos.transpose(1,2).reshape([1, -1, 16, 16, 16])
        x_pos = F.interpolate(x_pos, scale_factor = 2, mode='trilinear', align_corners=True)
        head_pos = F.interpolate(self.head(x_pos), scale_factor = 2, mode='trilinear', align_corners=True)
        return head_pos

class TaskPromptEncoder_Grade(nn.Module):
    def __init__(
            self,
            embedding_dim=256,
            mlp_dim=2048,
            mla_channels=256, 
            mlahead_channels=128, 
            num_heads=8
    ):
        super().__init__()
        self.cls_embedding_0 = nn.parameter.Parameter(torch.randn(1, 4096, 256))
        self.cls_embedding_1 = nn.parameter.Parameter(torch.randn(1, 4096, 256))
        self.attn = Attention_Grade(embedding_dim, num_heads)
        self.mlp = MLPBlock(embedding_dim, mlp_dim, nn.ReLU)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.head = nn.Sequential(nn.Conv3d(mla_channels, mlahead_channels, 3, padding=1, bias=False),
                     nn.InstanceNorm3d(mlahead_channels),
                     nn.ReLU(),
                     nn.Conv3d(mlahead_channels, mlahead_channels, 3, padding=1, bias=False),
                     nn.InstanceNorm3d(mlahead_channels),
                     nn.ReLU())
        
    def forward(self, x, i):
        x = F.interpolate(x, scale_factor = 0.5, mode='trilinear', align_corners=True)
        x = x.flatten(2).permute(0, 2, 1)
        candidates = [self.cls_embedding_0, self.cls_embedding_1]
        
        cls_embedding = candidates[i] 
        x_pos = self.attn(q=x, k=x, v=cls_embedding, i=i)
        x_pos = x + x_pos
        x_pos = x_pos + self.mlp(self.norm1(x_pos))
        x_pos = self.norm2(x_pos)
        x_pos = x_pos.transpose(1,2).reshape([1, -1, 16, 16, 16])
        x_pos = F.interpolate(x_pos, scale_factor = 2, mode='trilinear', align_corners=True)
        head_pos = F.interpolate(self.head(x_pos), scale_factor = 2, mode='trilinear', align_corners=True)
        return head_pos

class TaskPromptEncoder_Malignancy(nn.Module):
    def __init__(
            self,
            embedding_dim=256,
            mlp_dim=2048,
            mla_channels=256, 
            mlahead_channels=128, 
            num_heads=8
    ):
        super().__init__()
        self.cls_embedding_pos = nn.parameter.Parameter(torch.randn(1, 256, 16, 16, 16))
        self.cls_embedding_neg = nn.parameter.Parameter(torch.randn(1, 256, 16, 16, 16))
        self.attn = Attention_Malignancy(embedding_dim, num_heads)
        self.mlp = MLPBlock(embedding_dim, mlp_dim, nn.ReLU)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.head = nn.Sequential(nn.Conv3d(mla_channels, mlahead_channels, 3, padding=1, bias=False),
                     nn.InstanceNorm3d(mlahead_channels),
                     nn.ReLU(),
                     nn.Conv3d(mlahead_channels, mlahead_channels, 3, padding=1, bias=False),
                     nn.InstanceNorm3d(mlahead_channels),
                     nn.ReLU())
        
    def forward(self, x, pos_embed=None, neg_embed= None):
        x = F.interpolate(x, scale_factor = 0.5, mode='trilinear', align_corners=True)
        x = x.flatten(2).permute(0, 2, 1)
        if pos_embed!=None:
            cls_embedding_pos = pos_embed
            cls_embedding_neg = neg_embed
        else:
            cls_embedding_pos = self.cls_embedding_pos.flatten(2).permute(0, 2, 1)
            cls_embedding_neg = self.cls_embedding_neg.flatten(2).permute(0, 2, 1)

        x_pos, x_neg = self.attn(q=x, k=x, v_pos=cls_embedding_pos, v_neg = cls_embedding_neg)

        x_pos = x + x_pos
        x_pos = x_pos + self.mlp(self.norm1(x_pos))
        x_pos = self.norm2(x_pos)
        x_pos = x_pos.transpose(1,2).reshape([1, -1, 16, 16, 16])
        x_pos = F.interpolate(x_pos, scale_factor = 2, mode='trilinear', align_corners=True)
        head_pos = F.interpolate(self.head(x_pos), scale_factor = 2, mode='trilinear', align_corners=True)
                
        x_neg = x + x_neg
        x_neg = x_neg + self.mlp(self.norm1(x_neg))
        x_neg = self.norm2(x_neg)
        x_neg = x_neg.transpose(1,2).reshape([1, -1, 16, 16, 16])
        x_neg = F.interpolate(x_neg, scale_factor = 2, mode='trilinear', align_corners=True)
        head_neg = F.interpolate(self.head(x_neg), scale_factor = 2, mode='trilinear', align_corners=True)
        
        return head_pos, head_neg #x_pos, x_neg #head_pos, head_neg

class TaskPromptEncoder_Malignancy_s2(nn.Module):
    def __init__(
            self,
            embedding_dim=256,
            mlp_dim=2048,
            mla_channels=256, 
            mlahead_channels=128, 
            num_heads=8
    ):
        super().__init__()
        self.cls_embedding_pos = nn.parameter.Parameter(torch.randn(1, 256, 16, 16, 16))
        self.cls_embedding_neg = nn.parameter.Parameter(torch.randn(1, 256, 16, 16, 16))
        self.attn = Attention_Malignancy(embedding_dim, num_heads)
        self.mlp = MLPBlock(embedding_dim, mlp_dim, nn.ReLU)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.head = nn.Sequential(nn.Conv3d(mla_channels, mlahead_channels, 3, padding=1, bias=False),
                     nn.InstanceNorm3d(mlahead_channels),
                     nn.ReLU(),
                     nn.Conv3d(mlahead_channels, mlahead_channels, 3, padding=1, bias=False),
                     nn.InstanceNorm3d(mlahead_channels),
                     nn.ReLU())
        
    def forward(self, x, pos_embed=None, neg_embed= None):
        x = F.interpolate(x, scale_factor = 0.5, mode='trilinear', align_corners=True)
        x = x.flatten(2).permute(0, 2, 1)
        if pos_embed!=None:
            cls_embedding_pos = pos_embed
            x_pos, _ = self.attn(q=x, k=x, v_pos=cls_embedding_pos, v_neg = None)
            x_pos = x + x_pos
            x_pos = x_pos + self.mlp(self.norm1(x_pos))
            x_pos = self.norm2(x_pos)
            x_pos = x_pos.transpose(1,2).reshape([1, -1, 16, 16, 16])
            x_pos = F.interpolate(x_pos, scale_factor = 2, mode='trilinear', align_corners=True)
            head_pos = F.interpolate(self.head(x_pos), scale_factor = 2, mode='trilinear', align_corners=True)
            return head_pos
        if neg_embed!=None:                   
            cls_embedding_neg = neg_embed
            _, x_neg = self.attn(q=x, k=x, v_pos=None, v_neg = cls_embedding_neg)
            x_neg = x + x_neg
            x_neg = x_neg + self.mlp(self.norm1(x_neg))
            x_neg = self.norm2(x_neg)
            x_neg = x_neg.transpose(1,2).reshape([1, -1, 16, 16, 16])
            x_neg = F.interpolate(x_neg, scale_factor = 2, mode='trilinear', align_corners=True)
            head_neg = F.interpolate(self.head(x_neg), scale_factor = 2, mode='trilinear', align_corners=True)
            return head_neg 

class PromptEncoder(nn.Module):
    def __init__(
        self,
        *,
        transformer: nn.Module,
        num_pos_feats: int = 128,
        mask_prompt = False
    ) -> None:
        super().__init__()
        self.transformer = transformer
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
             torch.randn((3, num_pos_feats)),
        )
        self.mask_prompt = mask_prompt
        if mask_prompt:
            self.default_prompt = nn.parameter.Parameter(torch.randn(1, 256, 32, 32, 32))
            self.mask_encoder = nn.Sequential(
            nn.Conv3d(1, 256 // 4, kernel_size=3, stride=3),
            LayerNorm3d(256 // 4),
            nn.GELU(),
            nn.Conv3d(256 // 4, 256, kernel_size=3, padding = 1, stride=1),
            LayerNorm3d(256),
            nn.GELU(),
            nn.Conv3d(256, 256, kernel_size=1),
            )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        point_coord,
        img_size = [512, 512, 32],
        feat_size = [32, 32, 32]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.
        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.
        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        image_pe = self.get_img_pe(feat_size, device=image_embeddings.device).detach()
        '''
        if self.mask_prompt:
            if masks == None:
                image_embeddings += self.default_prompt
            else:
                image_embeddings += self.mask_encoder(masks)
        '''
        point_coord[:, :, 0] = (point_coord[:, :, 0]+0.5) * 2 / img_size[2] - 1
        point_coord[:, :, 1] = (point_coord[:, :, 1]+0.5) * 2 / img_size[1] - 1
        point_coord[:, :, 2] = (point_coord[:, :, 2]+0.5) * 2 / img_size[0] - 1
        point_coord = point_coord.reshape(1,1,1,-1,3)
        features = self.transformer(image_embeddings, image_pe, point_coord)
        features = features.transpose(1,2).reshape([1, -1] + feat_size)

        return features

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords * 3 / 2
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        coords[:, :, 2] = coords[:, :, 2] / image_size[2]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C

    def get_img_pe(self, size: Tuple[int, int], device) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w, d = size
        grid = torch.ones((h, w, d), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        z_embed = grid.cumsum(dim=2) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w
        z_embed = z_embed / d

        pe = self._pe_encoding(torch.stack([x_embed, y_embed, z_embed], dim=-1))
        return pe.permute(3, 0, 1, 2).unsqueeze(0)  # C x D X H x W
