#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#
# ============================================================
# architecture.py — SimpleFold 的核心模型结构定义
#
# 模型整体架构（从输入到输出）：
#
#   加噪坐标 ──→ Fourier Position Encoding ──→ ┐
#   原子特征 ──→ 特征拼接 + MLP ──────────────→─┤
#                         ↓                     │
#                   [Atom Encoder]              │  ← 原子级 Transformer
#                         ↓                     │
#            原子 → 残基 聚合 (bmm) ────────────→┤
#                         ↓                     │
#                   ESM 序列嵌入 ──→ 拼接 ─────→─┤
#                         ↓                     │
#                   [Trunk] (DiT)  ←───────────→┤  ← 残基级 Diffusion Transformer
#                         ↓                     │
#            残基 → 原子 广播 (bmm) + 跳跃连接 ─→┤
#                         ↓                     │
#                   [Atom Decoder]              │  ← 原子级 Transformer
#                         ↓                     │
#                FinalLayer (AdaLN + Linear) ───→┤
#                         ↓                     │
#                predict_velocity (B, N, 3) ────→┘
#
# ============================================================

import math
import torch
from torch import nn
from torch.nn import functional as F

from model.torch.layers import FinalLayer, ConditionEmbedder
from utils.esm_utils import esm_model_dict


class FoldingDiT(nn.Module):
    """SimpleFold 的核心扩散 Transformer 模型（Folding DiT）。

    这是一个 Flow Matching 去噪网络，架构采用了"原子级 → 残基级 → 原子级"的 U-Net 式设计：

    1. Atom Encoder（原子编码器）：将每个原子的特征编码到原子级隐空间
    2. 原子 → 残基聚合（attention-based pooling）：通过 atom_to_token 矩阵平均聚合
    3. Trunk（残基级主干网络）：DiTBlock 堆叠，在残基级别做自注意力
    4. 残基 → 原子广播 + 跳跃连接：通过 atom_to_token 矩阵广播回原子空间
    5. Atom Decoder（原子解码器）：在原子级别做最终的精炼
    6. FinalLayer：adaLN-Zero 调节 + 线性投影出 (dx, dy, dz)

    整个网络预测的是 velocity field v_t（流匹配的目标），而不是直接预测去噪坐标。
    """

    def __init__(
        self,
        trunk,
        time_embedder,
        aminoacid_pos_embedder,
        pos_embedder,
        atom_encoder_transformer,
        atom_decoder_transformer,
        hidden_size=1152,
        num_heads=16,
        atom_num_heads=4,
        output_channels=3,
        atom_hidden_size_enc=256,
        atom_hidden_size_dec=256,
        atom_n_queries_enc=32,
        atom_n_keys_enc=128,
        atom_n_queries_dec=32,
        atom_n_keys_dec=128,
        esm_model="esm2_3B",
        esm_dropout_prob=0.0,
        use_atom_mask=False,
        use_length_condition=True,
    ):
        """初始化 FoldingDiT 模型。

        Args:
            trunk: 残基级别的主干 Transformer（HomogenTrunk[DiTBlock]）
            time_embedder: 时间步 t 的嵌入层（TimestepEmbedder）
            aminoacid_pos_embedder: 氨基酸序列位置编码（AbsolutePositionEncoding）
            pos_embedder: 3D 坐标傅立叶位置编码（FourierPositionEncoding）
            atom_encoder_transformer: 原子编码器 Transformer
            atom_decoder_transformer: 原子解码器 Transformer
            hidden_size: 残基级别隐层维度（控制模型容量）
            num_heads: 残基级别注意力头数
            atom_num_heads: 原子级别注意力头数
            output_channels: 输出通道数（3 = x, y, z）
            atom_hidden_size_enc: 原子编码器隐层维度
            atom_hidden_size_dec: 原子解码器隐层维度
            atom_n_queries_enc/dec: 原子注意力窗口中的 query 数（局部注意力）
            atom_n_keys_enc/dec: 原子注意力窗口中的 key/value 数
            esm_model: ESM 模型名称（用于获取 ESM 输出维度等元数据）
            esm_dropout_prob: ESM 特征 dropout 概率（用于 classifier-free guidance）
            use_atom_mask: 是否使用原子掩码
            use_length_condition: 是否使用序列长度作为条件特征
        """
        super().__init__()
        self.pos_embedder = pos_embedder
        pos_embed_channels = pos_embedder.embed_dim
        self.aminoacid_pos_embedder = aminoacid_pos_embedder
        aminoacid_pos_embed_channels = aminoacid_pos_embedder.embed_dim

        self.time_embedder = time_embedder

        self.atom_encoder_transformer = atom_encoder_transformer
        self.atom_decoder_transformer = atom_decoder_transformer

        self.trunk = trunk

        self.hidden_size = hidden_size
        self.output_channels = output_channels
        self.num_heads = num_heads
        self.atom_num_heads = atom_num_heads
        self.use_atom_mask = use_atom_mask
        self.esm_dropout_prob = esm_dropout_prob
        self.use_length_condition = use_length_condition

        # 从 ESM 模型配置中读取维度信息
        esm_s_dim = esm_model_dict[esm_model]["esm_s_dim"]
        esm_num_layers = esm_model_dict[esm_model]["esm_num_layers"]

        self.atom_hidden_size_enc = atom_hidden_size_enc
        self.atom_hidden_size_dec = atom_hidden_size_dec
        self.atom_n_queries_enc = atom_n_queries_enc
        self.atom_n_keys_enc = atom_n_keys_enc
        self.atom_n_queries_dec = atom_n_queries_dec
        self.atom_n_keys_dec = atom_n_keys_dec

        # ========== 原子特征投影层 ==========
        # 原子特征的拼接维度：
        #   pos_embed_channels (Fourier 编码) +
        #   aminoacid_pos_embed_channels (序列位置编码) +
        #   427 (one-hot 残基类型 33 + 分子类型 4 + pocket 4 + 电荷 1 +
        #         掩码 1 + 元素 128 + 原子名 256)
        atom_feat_dim = pos_embed_channels + aminoacid_pos_embed_channels + 427
        self.atom_feat_proj = nn.Sequential(
            nn.Linear(atom_feat_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
        )
        # 加噪坐标的位置编码投影到隐空间
        self.atom_pos_proj = nn.Linear(pos_embed_channels, hidden_size, bias=False)

        # 可选的序列长度条件嵌入
        if self.use_length_condition:
            self.length_embedder = nn.Sequential(
                nn.Linear(1, hidden_size, bias=False),
                nn.LayerNorm(hidden_size),
            )

        # 原子特征（768D）与坐标编码（768D）拼接后投影回 768D
        self.atom_in_proj = nn.Linear(hidden_size * 2, hidden_size, bias=False)

        # ========== ESM 特征融合层 ==========
        # esm_s_combine: 可学习的 ESM 各层加权融合权重（softmax 归一化）
        self.esm_s_combine = nn.Parameter(torch.zeros(esm_num_layers))
        # esm_s_proj: 将 ESM 序列嵌入投影到模型隐空间（含 conditioned dropout）
        self.esm_s_proj = ConditionEmbedder(
            input_dim=esm_s_dim,
            hidden_size=hidden_size,
            dropout_prob=self.esm_dropout_prob,
        )
        # 原子聚合后的 latent 与 ESM 嵌入拼接后投影融合
        latent_cat_dim = hidden_size * 2
        self.esm_cat_proj = nn.Linear(latent_cat_dim, hidden_size)

        # ========== 原子 ↔ 残基之间转换的投影层 ==========
        # 原子编码器：将残基级条件嵌入下采样到原子级维度
        self.context2atom_proj = nn.Sequential(
            nn.Linear(hidden_size, self.atom_hidden_size_enc),
            nn.LayerNorm(self.atom_hidden_size_enc),
        )
        # 原子编码后上采样回残基级维度
        self.atom2latent_proj = nn.Sequential(
            nn.Linear(self.atom_hidden_size_enc, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        # 原子编码器的条件投影
        self.atom_enc_cond_proj = nn.Sequential(
            nn.Linear(hidden_size, self.atom_hidden_size_enc),
            nn.LayerNorm(self.atom_hidden_size_enc),
        )
        # 原子解码器的条件投影
        self.atom_dec_cond_proj = nn.Sequential(
            nn.Linear(hidden_size, self.atom_hidden_size_dec),
            nn.LayerNorm(self.atom_hidden_size_dec),
        )

        # 残基级 latent → 原子解码器输入的投影（含 SiLU 激活）
        self.latent2atom_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, self.atom_hidden_size_dec),
        )

        # 最终输出层：adaLN-Zero 调节 + 线性投影 => (dx, dy, dz)
        self.final_layer = FinalLayer(
            self.atom_hidden_size_dec,
            output_channels,
            c_dim=hidden_size
        )

    def create_local_attn_bias(
        self, n: int, n_queries: int, n_keys: int, inf: float = 1e10, device: torch.device = None
    ) -> torch.Tensor:
        """创建局部注意力偏置（local attention bias）。

        由于蛋白质中的原子数量可能很大（几千个），全局自注意力的 O(N²) 计算不可行。
        这里使用局部注意力窗口：每个 query 只能关注其相邻的 n_keys 个 key/value。

        实现方式：生成一个对角线风格的注意力掩码，每个 query 窗口大小为 n_queries，
                可关注的 key 窗口大小为 n_keys（以 query 窗口为中心）。

        Args:
            n (int): query 序列长度（原子数 N）
            n_queries (int): 每个 query 窗口中的 query 数量
            n_keys (int): 每个 query 窗口能关注的 key 数量（>= n_queries）
            inf (float, optional): 用于掩码的大负数. Defaults to 1e10.
            device (torch.device, optional): 设备. Defaults to None.

        Returns:
            torch.Tensor: 局部注意力偏置矩阵 [n, n]，
                          允许关注的位置为 0，禁止的为 -inf
        """
        n_trunks = int(math.ceil(n / n_queries))
        padded_n = n_trunks * n_queries
        attn_mask = torch.zeros(padded_n, padded_n, device=device)
        for block_index in range(0, n_trunks):
            i = block_index * n_queries
            j1 = max(0, n_queries * block_index - (n_keys - n_queries) // 2)
            j2 = n_queries * block_index + (n_queries + n_keys) // 2
            attn_mask[i : i + n_queries, j1:j2] = 1.0
        attn_bias = (1 - attn_mask) * -inf
        return attn_bias.to(device=device)[:n, :n]

    def create_atom_attn_mask(
        self,
        feats,
        natoms,
        atom_n_queries=None,
        atom_n_keys=None,
        inf: float = 1e10
    ) -> torch.Tensor:
        """为原子编码器/解码器创建局部注意力掩码。

        如果提供了 query/key 窗口大小，则创建局部注意力偏置；
        否则返回 None（使用全局注意力）。
        """
        if atom_n_queries is not None and atom_n_keys is not None:
            atom_attn_mask = self.create_local_attn_bias(
                n=natoms,
                n_queries=atom_n_queries,
                n_keys=atom_n_keys,
                device=feats["ref_pos"].device,
                inf=inf,
            )
        else:
            atom_attn_mask = None

        return atom_attn_mask

    def forward(self, noised_pos, t, feats, self_cond=None):
        """FoldingDiT 的前向传播。

        流程：原子级 → 残基级（聚合）→ 残基级（Trunk）→ 原子级（广播）→ 输出

        Args:
            noised_pos: 加噪的原子坐标 [B, N, 3]（在扩散/流匹配过程中的中间状态）
            t: 时间步 [B]（0=完全噪声, 1=完全数据）
            feats: 特征字典，包含：
                ref_pos: 参考位置 [B, N, 3]
                mol_type: 分子类型 [B, M, 4]（蛋白/DNA/RNA/小分子）
                res_type: 残基类型 [B, M, 33]（one-hot 编码的 20 种氨基酸 + ...）
                atom_to_token: 原子到残基的映射矩阵 [B, N, M]（稀疏的分配矩阵）
                ref_charge: 原子电荷 [B, N]
                atom_pad_mask: 原子填充掩码 [B, N]
                ref_element: 原子元素类型 [B, N, 128]
                ref_atom_name_chars: 原子名字符编码 [B, N, 256]
                esm_s: ESM 所有层的序列嵌入 [B, M, esm_num_layers, esm_s_dim]
                residue_index/entity_id/asym_id/sym_id: 用于轴向 RoPE 的位置标识符
                ref_space_uid: 参考空间 UID [B, N]
                atom_to_token_idx: 原子所属的残基索引 [B, N]
                pocket_feature: pocket 特征 [B, M, 4]
                max_num_tokens: 序列最大长度 [B]（用于 length conditioning）
            self_cond: 自条件化的 latent（当前未使用）

        Returns:
            dict: {
                "predict_velocity": [B, N, 3] 预测的速度场（流匹配目标），
                "latent": [B, M, D] 残基级别的隐向量（用于 pLDDT 预测）
            }
        """
        B, N, _ = feats["ref_pos"].shape  # B=batch, N=原子总数
        M = feats["mol_type"].shape[1]     # M=残基/令牌总数
        atom_to_token = feats["atom_to_token"].float()  # [B, N, M] 原子→残基分配矩阵
        atom_to_token_idx = feats["atom_to_token_idx"]
        ref_space_uid = feats["ref_space_uid"]

        # ================================================================
        # 1. 创建原子级别的局部注意力掩码
        # ================================================================
        # 蛋白质的原子数通常很大（数千），所以使用局部注意力来节省计算。
        # 每个原子只能关注其窗口内的相邻原子。
        atom_attn_mask_enc = self.create_atom_attn_mask(
            feats,
            natoms=N,
            atom_n_queries=self.atom_n_queries_enc,
            atom_n_keys=self.atom_n_keys_enc,
        )
        atom_attn_mask_dec = self.create_atom_attn_mask(
            feats,
            natoms=N,
            atom_n_queries=self.atom_n_queries_dec,
            atom_n_keys=self.atom_n_keys_dec,
        )

        # ================================================================
        # 2. 创建条件嵌入（time + length）
        # ================================================================
        # c_emb 会通过 adaLN-Zero 注入到每个 DiTBlock 中
        c_emb = self.time_embedder(t)  # (B, D)
        if self.use_length_condition:
            length = feats["max_num_tokens"].float().unsqueeze(-1)
            c_emb = c_emb + self.length_embedder(torch.log(length))

        # ================================================================
        # 3. 构建原子特征
        # ================================================================
        mol_type = feats["mol_type"]
        mol_type = F.one_hot(mol_type, num_classes=4).float()        # [B, M, 4]
        res_type = feats["res_type"].float()                         # [B, M, 33]
        pocket_feature = feats["pocket_feature"].float()             # [B, M, 4]
        res_feat = torch.cat(
            [mol_type, res_type, pocket_feature],
            dim=-1,
        )                                                             # [B, M, 41]
        # 用 atom_to_token 将残基特征广播到每个原子上
        atom_feat_from_res = torch.bmm(atom_to_token, res_feat)      # [B, N, 41]

        # 氨基酸级别的绝对序列位置编码
        atom_res_pos = self.aminoacid_pos_embedder(
            pos=atom_to_token_idx.unsqueeze(-1).float()
        )

        # 参考坐标的傅立叶位置编码（3D 结构的空间信息）
        ref_pos_emb = self.pos_embedder(pos=feats["ref_pos"])

        # 拼接所有原子特征
        atom_feat = torch.cat(
            [
                ref_pos_emb,                                        # (B, N, PD1) 空间位置编码
                atom_feat_from_res,                                 # (B, N, 41) 残基派生特征
                atom_res_pos,                                       # (B, N, PD2) 序列位置编码
                feats["ref_charge"].unsqueeze(-1),                  # (B, N, 1) 原子电荷
                feats["atom_pad_mask"].unsqueeze(-1),               # (B, N, 1) 有效原子掩码
                feats["ref_element"],                               # (B, N, 128) 元素类型 one-hot
                feats["ref_atom_name_chars"].reshape(B, N, 4 * 64), # (B, N, 256) 原子名编码
            ],
            dim=-1,
        )                                                            # (B, N, PD1+PD2+427)
        atom_feat = self.atom_feat_proj(atom_feat)                  # (B, N, D) 投影到隐空间

        # ================================================================
        # 4. 编码加噪坐标的位置信息
        # ================================================================
        # 对加噪的坐标同样做傅立叶编码，然后投影到隐空间
        atom_coord = self.pos_embedder(pos=noised_pos)              # (B, N, PD1)
        atom_coord = self.atom_pos_proj(atom_coord)                 # (B, N, D)

        # 将原子特征与坐标特征拼接融合
        atom_in = torch.cat([atom_feat, atom_coord], dim=-1)
        atom_in = self.atom_in_proj(atom_in)                        # (B, N, D)

        # ================================================================
        # 5. 准备轴向 RoPE 位置编码的输入
        # ================================================================
        # 原子级别的 RoPE：由 ref_space_uid + ref_pos（3D 坐标）组成
        atom_pe_pos = torch.cat(
            [
                ref_space_uid.unsqueeze(-1).float(),                 # (B, N, 1) 空间 UID
                feats["ref_pos"],                                    # (B, N, 3) 3D 坐标
            ],
            dim=-1,
        )                                                             # (B, N, 4)
        # 残基级别的 RoPE：由 residue_index + entity_id + asym_id + sym_id 组成
        token_pe_pos = torch.cat(
            [
                feats["residue_index"].unsqueeze(-1).float(),        # (B, M, 1) 残基序号
                feats["entity_id"].unsqueeze(-1).float(),            # (B, M, 1) 实体 ID
                feats["asym_id"].unsqueeze(-1).float(),              # (B, M, 1) 不对称单元 ID
                feats["sym_id"].unsqueeze(-1).float(),               # (B, M, 1) 对称 ID
            ],
            dim=-1,
        )                                                             # (B, M, 4)

        # ================================================================
        # 6. 原子编码器（Atom Encoder）
        # ================================================================
        # 在原子级别做局部自注意力，提取每个原子的上下文表示
        # 这是"从原子看到局部结构"
        atom_c_emb_enc = self.atom_enc_cond_proj(c_emb)              # 条件嵌入投影到原子级维度
        atom_latent = self.context2atom_proj(atom_in)                # 输入投影到原子级隐空间
        atom_latent = self.atom_encoder_transformer(
            latents=atom_latent,
            c=atom_c_emb_enc,
            attention_mask=atom_attn_mask_enc,
            pos=atom_pe_pos,
        )
        atom_latent = self.atom2latent_proj(atom_latent)             # 上采样回残基级维度

        # ================================================================
        # 7. 原子 → 残基 聚合（Perceiver-style pooling）
        # ================================================================
        # 使用 atom_to_token 矩阵，将原子级别的表示平均聚合到残基级别。
        # 每个残基可能有多个原子（如蛋白质的 CA/C/N/O...），这里取平均。
        atom_to_token_mean = atom_to_token / (
            atom_to_token.sum(dim=1, keepdim=True) + 1e-6
        )
        latent = torch.bmm(atom_to_token_mean.transpose(1, 2), atom_latent)
        assert latent.shape[1] == M

        # ================================================================
        # 8. ESM 特征融合
        # ================================================================
        # ESM 提供了强大的序列上下文嵌入。
        # esm_s_combine: 可学习的各层融合权重（softmax 归一化后加权平均所有层）
        esm_s = (self.esm_s_combine.softmax(0).unsqueeze(0) @ feats['esm_s']).squeeze(2)
        force_drop_ids = feats.get("force_drop_ids", None)
        esm_emb = self.esm_s_proj(esm_s, self.training, force_drop_ids)
        assert esm_emb.shape[1] == latent.shape[1]

        # 将原子聚合的 latent 与 ESM 嵌入拼接并投影融合
        latent = self.esm_cat_proj(torch.cat([latent, esm_emb], dim=-1))

        # ================================================================
        # 9. 残基主干网络（Trunk）
        # ================================================================
        # 这是模型的核心：在残基级别做深度 Transformer 处理（DiTBlock 堆叠）。
        # 这一层捕捉残基之间的远程相互作用（这是蛋白质折叠的关键）。
        latent = self.trunk(
            latents=latent,
            c=c_emb,
            attention_mask=None,    # 残基级别通常使用全局注意力（M << N）
            pos=token_pe_pos,
        )

        # ================================================================
        # 10. 残基 → 原子 广播 + 跳跃连接
        # ================================================================
        # 将残基级别的表示通过 atom_to_token 矩阵广播回每个原子。
        # 同时加上来自原子编码器的跳跃连接（类似 U-Net 结构）。
        output = torch.bmm(atom_to_token, latent)
        assert output.shape[1] == N

        # 跳跃连接：原子编码器的输出直接添加到原子解码器输入
        output = output + atom_latent
        output = self.latent2atom_proj(output)                       # 投影到原子解码器维度

        # ================================================================
        # 11. 原子解码器 + 最终输出层
        # ================================================================
        # 原子解码器在原子级别做进一步的精炼
        atom_c_emb_dec = self.atom_dec_cond_proj(c_emb)
        output = self.atom_decoder_transformer(
            latents=output,
            c=atom_c_emb_dec,
            attention_mask=atom_attn_mask_dec,
            pos=atom_pe_pos,
        )
        # FinalLayer: adaLN-Zero 调节 + 线性层 => 预测 velocity
        output = self.final_layer(output, c=c_emb)

        return {
            "predict_velocity": output,  # [B, N, 3] 预测的速度场
            "latent": latent,            # [B, M, D] 残基级隐向量（用于 pLDDT 头）
        }
