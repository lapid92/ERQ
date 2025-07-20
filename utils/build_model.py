import numpy as np
from types import MethodType
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models import vision_transformer
from timm.models.vision_transformer import Attention, Block, VisionTransformer
from timm.models.swin_transformer import WindowAttention

from utils.hadamard_utils import random_hadamard_matrix

def vision_transformer_block_forward(self, x):
    x = self.shortcut1(x + self.drop_path(self.attn(self.norm1(x))))
    x = self.shortcut2(x + self.drop_path(self.mlp(self.norm2(x))))
    return x

def forward_features(self, x):
    x = self.patch_embed(x)
    cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
    if self.dist_token is None:
        x = torch.cat((cls_token, x), dim=1)
    else:
        x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
    x = self.pos_drop(x + self.pos_embed)
    x = self.embed_act_quant(x)
    x = self.blocks(x)
    x = self.norm_head_act_quant(self.norm(x))
    if self.dist_token is None:
        return self.pre_logits(x[:, 0])
    else:
        return x[:, 0], x[:, 1]


def attention_forward(self, x):
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

    # attn = (q @ k.transpose(-2, -1)) * self.scale
    attn = self.matmul1(q, k.transpose(-2, -1)) * self.scale
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)
    # del q, k

    # x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    x = self.matmul2(attn, v).transpose(1, 2).reshape(B, N, C)
    # del attn, v
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


def window_attention_forward(self, x, mask=None):
    B_, N, C = x.shape
    qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

    q = q * self.scale
    # attn = (q @ k.transpose(-2, -1))
    attn = self.matmul1(q, k.transpose(-2, -1))

    relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
        self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
    relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    attn = attn + relative_position_bias.unsqueeze(0)

    if mask is not None:
        nW = mask.shape[0]
        attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)
    else:
        attn = self.softmax(attn)

    attn = self.attn_drop(attn)

    # x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
    x = self.matmul2(attn, v).transpose(1, 2).reshape(B_, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


class ActQuant(nn.Module):
    def forward(self, x):
        return x

class MatMul(nn.Module):
    def forward(self, A, B):
        return A @ B


def build_model(name):
    """
    Get a vision transformer model.
    This will replace matrix multiplication operations with matmul modules in the model.
    Currently support almost all models in timm.models.transformers, including:
    - vit_tiny/small/base/large_patch16/patch32_224/384,
    - deit_tiny/small/base(_distilled)_patch16_224,
    - deit_base(_distilled)_patch16_384,
    - swin_tiny/small/base/large_patch4_window7_224,
    - swin_base/large_patch4_window12_384
    These models are finetuned on imagenet-1k and should use ViTImageNetLoaderGenerator
    for calibration and testing.
    """
    model = timm.create_model(name, pretrained=True)

    # for module in model.modules():
    for name, module in model.named_modules():
        if isinstance(module, Attention):
            setattr(module, "matmul1", MatMul())
            setattr(module, "matmul2", MatMul())
            module.forward = MethodType(attention_forward, module)
        if isinstance(module, WindowAttention):
            setattr(module, "matmul1", MatMul())
            setattr(module, "matmul2", MatMul())
            module.forward = MethodType(window_attention_forward, module)
        if isinstance(module, Block):
            setattr(module, "shortcut1", ActQuant())
            setattr(module, "shortcut2", ActQuant())
            module.forward = MethodType(vision_transformer_block_forward, module)
        if isinstance(module, VisionTransformer):
            setattr(module, "embed_act_quant", ActQuant())
            setattr(module, "norm_head_act_quant", ActQuant())
            module.forward_features = MethodType(forward_features, module)

    return model


def random_orthogonal_matrix(size, device):
    torch.cuda.empty_cache()
    random_matrix = torch.randn(size, size).to(device)
    q, r = torch.linalg.qr(random_matrix)
    q *= torch.sign(torch.diag(r)).unsqueeze(0)
    return q


def get_orthogonal_matrix(size, mode="hadamard", device='cuda'):
    if mode == "random":
        return random_orthogonal_matrix(size, device)
    elif mode == "hadamard":
        return random_hadamard_matrix(size, device)
    return False


def rotate_model(model, rotation_type):
    device = model.head.weight.device
    hidden_size = model.embed_dim
    R = get_orthogonal_matrix(size=hidden_size, mode=rotation_type, device=device)

    # For LN folding
    C = torch.eye(hidden_size) - torch.ones((hidden_size, hidden_size)) / hidden_size
    C = C.to(device)
    R1 = torch.matmul(R, C)

    module_dict = {}
    _modules = dict(model.named_modules())
    for name, module in _modules.items():
        module_dict[name] = module
        idx = name.rfind('.')
        if idx == -1:
            idx = 0
        father_name = name[:idx]
        if father_name in module_dict:
            father_module = module_dict[father_name]
        else:
            raise RuntimeError(f"father module {father_name} not found")

        if isinstance(module, nn.LayerNorm):
            # Check cases
            norm_type = name.split('.')[-1]
            # Attention
            if norm_type == 'norm1':
                input_linear_module = model.get_submodule(father_name + '.attn.qkv')
                if input_linear_module.bias is not None:
                    input_linear_module.bias = nn.Parameter(input_linear_module.bias + input_linear_module.weight @ module.bias)
                module.bias = nn.Parameter(torch.zeros_like(module.bias))

                input_linear_module.weight = nn.Parameter(input_linear_module.weight * module.weight)
                module.weight = nn.Parameter(torch.ones_like(module.weight))

                # Step 2.b
                # TODO:
                # apply from outside flag
                # setattr(model, name, nn.RMSNorm(hidden_size))

                # Step 2.c - Fold R.T into LINEAR_IN
                input_linear_module.weight = nn.Parameter(torch.matmul(input_linear_module.weight, R.T))

                output_linear_module = model.get_submodule(father_name + '.attn.proj')

                output_linear_module.weight = nn.Parameter(torch.matmul(R, output_linear_module.weight))

                if output_linear_module.bias is not None:
                    output_linear_module.bias = nn.Parameter(torch.matmul(R, output_linear_module.bias))
            # MLP
            elif norm_type == 'norm2':
                input_linear_module = model.get_submodule(father_name + '.mlp.fc1')
                if input_linear_module.bias is not None:
                    input_linear_module.bias = nn.Parameter(input_linear_module.bias + input_linear_module.weight @ module.bias)
                module.bias = nn.Parameter(torch.zeros_like(module.bias))

                input_linear_module.weight = nn.Parameter(input_linear_module.weight * module.weight)
                module.weight = nn.Parameter(torch.ones_like(module.weight))

                # Step 2.b
                # TODO:
                # apply from outside flag
                # norm_node.layer_class = nn.RMSNorm
                # norm_node.framework_attr.pop('bias')
                # norm_node.weights.pop('bias')

                # Step 2.c - Fold R.T into LINEAR_IN
                input_linear_module.weight = nn.Parameter(torch.matmul(input_linear_module.weight, R.T))

                output_linear_module = model.get_submodule(father_name + '.mlp.fc2')

                output_linear_module.weight = nn.Parameter(torch.matmul(R, output_linear_module.weight))

                if output_linear_module.bias is not None:
                    output_linear_module.bias = nn.Parameter(torch.matmul(R, output_linear_module.bias))
            elif norm_type == 'norm':
                # TODO:
                # add R
                head_module = model.get_submodule('head')
                if head_module.bias is not None:
                    head_module.bias = nn.Parameter(head_module.bias + head_module.weight @ module.bias)
                module.bias = nn.Parameter(torch.zeros_like(module.bias))

                head_module.weight = nn.Parameter(head_module.weight * module.weight)
                module.weight = nn.Parameter(torch.ones_like(module.weight))

                # Step 2.b
                # TODO:
                # apply from outside flag
                # norm_node.layer_class = nn.RMSNorm
                # norm_node.framework_attr.pop('bias')
                # norm_node.weights.pop('bias')

                # Step 2.c - Fold R.T into LINEAR_IN
                head_module.weight = nn.Parameter(torch.matmul(head_module.weight, R.T))

        if isinstance(module, nn.Conv2d):
            # Rotate Conv
            folded_weight = torch.einsum('oi,icxy->ocxy', R1, module.weight)
            module.weight = nn.Parameter(folded_weight)

            # Rotate bias
            if module.bias is not None:
                module.bias = nn.Parameter(R1 @ module.bias)

            # Rotate cls_token
            if hasattr(model, 'cls_token') and model.cls_token is not None:
                model.cls_token = nn.Parameter(model.cls_token @ R1.T)

            # Rotate pos_embed
            if hasattr(model, 'pos_embed') and model.pos_embed is not None:
                model.pos_embed = nn.Parameter(model.pos_embed @ R1.T)

            # Rotate pos_embed
            if hasattr(model, 'absolute_pos_embed') and model.absolute_pos_embed is not None:
                model.absolute_pos_embed = nn.Parameter(model.absolute_pos_embed @ R1.T)
    # fix head
    # skin every LN
    # rotate every linear by name
    return model
