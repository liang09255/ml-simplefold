#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#
# ============================================================
# blocks.py — SimpleFold 的 Transformer 模块定义
#
# 包含三个核心模块：
#   1. DiTBlock          — 带 adaLN-Zero 调节的条件 Transformer 块
#   2. TransformerBlock  — 不带条件调节的普通 Transformer 块（用于 pLDDT 头）
#   3. HomogenTrunk      — 同类型 Block 的均匀堆叠容器
# ============================================================

import torch
from torch import nn
from timm.models.vision_transformer import Mlp
from model.torch.layers import modulate, SwiGLUFeedForward


class DiTBlock(nn.Module):
    """Diffusion Transformer Block (DiTBlock)，带 adaLN-Zero 条件调节。

    这是 SimpleFold 中最核心的 Transformer 模块，参考了 DiT（Diffusion Transformer）论文。
    核心创新：adaLN-Zero (Adaptive Layer Norm Zero) 条件调节机制。

    与标准 Transformer 的区别：
    - 用 adaLN 替代了固定的 LayerNorm：scale 和 shift 由条件嵌入 c 动态生成
    - 使用 gate 机制（zero-initialized）：每个子层输出先乘 gate 再加残差
    - gate 初始化为 0，确保训练初期所有层都是恒等映射，稳定训练

    结构：Norm(adaLN) → Self-Attention → Gate → Norm(adaLN) → FFN → Gate
    """

    def __init__(
        self,
        self_attention_layer,
        hidden_size,
        mlp_ratio=4.0,
        use_swiglu=True,
    ):
        """
        Args:
            self_attention_layer: 自注意力层的工厂函数（callable，返回 nn.Module）
            hidden_size: 隐层维度
            mlp_ratio: FFN 隐层维度 = hidden_size * mlp_ratio
            use_swiglu: 是否使用 SwiGLU 激活（比标准 GELU 更好）
        """
        super().__init__()
        # 注意力前的 LayerNorm（elementwise_affine=False 是因为 adaLN 提供缩放和平移）
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = self_attention_layer()

        # FFN 前的 LayerNorm
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        if use_swiglu:
            self.mlp = SwiGLUFeedForward(hidden_size, mlp_hidden_dim)
        else:
            approx_gelu = lambda: nn.GELU(approximate="tanh")
            self.mlp = Mlp(
                in_features=hidden_size,
                hidden_features=mlp_hidden_dim,
                act_layer=approx_gelu,
                drop=0,
            )

        # adaLN-Zero 调节网络：从条件 c 生成 6 个调制参数
        # 输出 6*hidden_size = [shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp]
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.initialize_weights()

    def initialize_weights(self):
        """初始化权重：注意头和 FFN 用 xavier_uniform，adaLN 的最后一层初始化为 0。"""
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Zero-out adaLN 的最后一层 Linear：使所有 gate=0, shift=0, scale=0
        # 这样训练初期相当于恒等映射，训练更稳定
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(
        self,
        latents,
        c,
        **kwargs,
    ):
        """前向传播。

        Args:
            latents: 输入的隐向量 (B, N, D)
            c: 条件嵌入 (B, D)，由 time_embedding + 其他条件组合而成
            **kwargs: 传递给自注意力的额外参数（如 pos 用于 RoPE, attention_mask 等）

        Returns:
            (B, N, D) 处理后的隐向量
        """
        # 从条件嵌入 c 中调制出 6 个参数
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )

        # 注意力子层：调制 LayerNorm → 自注意力 → gate 缩放 → 残差连接
        _latents = self.attn(
            modulate(self.norm1(latents), shift_msa, scale_msa), **kwargs
        )
        latents = latents + gate_msa.unsqueeze(1) * _latents

        # FFN 子层：调制 LayerNorm → FFN → gate 缩放 → 残差连接
        latents = latents + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(latents), shift_mlp, scale_mlp)
        )
        return latents


class TransformerBlock(nn.Module):
    """普通 Transformer Block（不带条件调节）。

    用于 pLDDT 置信度头等不需要扩散条件的地方。
    结构：LayerNorm → Self-Attention → 残差 → LayerNorm → FFN → 残差
    """

    def __init__(
        self,
        self_attention_layer,
        hidden_size,
        mlp_ratio=4.0,
        use_swiglu=False,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = self_attention_layer()
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        if use_swiglu:
            self.mlp = SwiGLUFeedForward(hidden_size, mlp_hidden_dim)
        else:
            self.mlp = Mlp(
                in_features=hidden_size,
                hidden_features=mlp_hidden_dim,
                act_layer=approx_gelu,
                drop=0,
            )

    def forward(
        self,
        latents,
        **kwargs,
    ):
        """不带条件的前向传播（直接用 LayerNorm）。

        注意：此模块与 DiTBlock 不同，不接收 c（条件）参数。
        也没有 adaLN 调制和 gate 机制。
        """
        _latents = self.attn(self.norm1(latents), **kwargs)
        latents = latents + _latents
        latents = latents + self.mlp(self.norm2(latents))
        return latents


class HomogenTrunk(nn.Module):
    """均匀堆叠容器：将同一类型的 Block 重复 depth 次。

    类似于 Transformer 的 Encoder，但所有块可以是同一配置实例化。
    用于：
    - 主干网络（Trunk）：DiTBlock × depth（如 8 层、36 层等）
    - 原子编码器/解码器：DiTBlock × depth（如 1 层、4 层等）
    - pLDDT 头：TransformerBlock × depth

    每个 block 是用 _partial_ 配置的工厂函数（实例化时传入 _partial_=True 的参数，
    然后在 __init__ 中调用 block() 生成实际模块）。
    """

    def __init__(self, block, depth):
        """
        Args:
            block: 可调用对象（工厂函数），每次调用返回一个新 Block 实例
            depth: 堆叠的 Block 数量
        """
        super().__init__()
        self.blocks = nn.ModuleList([block() for _ in range(depth)])

    def forward(self, latents, c, **kwargs):
        """逐层通过所有 Block。

        Args:
            latents: 输入的隐向量 (B, N, D)
            c: 条件嵌入 (B, D)，传递给每个 Block（如果是 DiTBlock 则使用，TransformerBlock 忽略）
            **kwargs: 额外参数（如 pos, attention_mask），传递给每个 Block
        """
        for i, block in enumerate(self.blocks):
            kwargs["layer_idx"] = i
            latents = block(latents=latents, c=c, **kwargs)
        return latents
