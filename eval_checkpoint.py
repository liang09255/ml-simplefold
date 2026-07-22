#!/usr/bin/env python3
"""
eval_checkpoint.py — 验证 checkpoint 能否做前向传播并评估指标
"""
import os, sys, json, warnings, argparse
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.insert(0, "src/simplefold")

import numpy as np
import torch
from pathlib import Path
from dataclasses import replace

device = torch.device("cpu")
print(f"设备: {device}\n")

from model.simplefold import SimpleFold
from processor.protein_processor import ProteinDataProcessor
from boltz_data_pipeline.tokenize.boltz_protein import BoltzTokenizer
from boltz_data_pipeline.feature.featurizer import BoltzFeaturizer
from boltz_data_pipeline.types import Record
from utils.datamodule_utils import load_input, collate


def compute_rmsd(pred, true, mask):
    p = (pred * mask.unsqueeze(-1)).cpu().numpy()
    t = (true * mask.unsqueeze(-1)).cpu().numpy()
    p = p[mask.cpu().numpy()]; t = t[mask.cpu().numpy()]
    if len(p) < 3: return 999.0
    pc = p - p.mean(0); tc = t - t.mean(0)
    u, s, vh = np.linalg.svd(pc.T @ tc)
    R = (u @ vh).T
    if np.linalg.det(R) < 0:
        u[:, -1] *= -1
        R = (u @ vh).T
    return float(np.sqrt(np.mean(np.sum((pc @ R - tc) ** 2, axis=-1))))


def compute_lddt(pred, true, mask, cutoff=15.0):
    p = (pred * mask.unsqueeze(-1)).cpu().numpy()
    t = (true * mask.unsqueeze(-1)).cpu().numpy()
    p = p[mask.cpu().numpy()]; t = t[mask.cpu().numpy()]
    if len(p) < 5: return 0.0
    pdm = np.sqrt(np.sum((p[:, None] - p[None, :]) ** 2, -1))
    tdm = np.sqrt(np.sum((t[:, None] - t[None, :]) ** 2, -1))
    pair = ~np.eye(len(p), dtype=bool) & (tdm < cutoff)
    if pair.sum() == 0: return 0.0
    diff = np.abs(pdm - tdm)
    score = 0.25 * ((diff < 0.5) + (diff < 1.0) + (diff < 2.0) + (diff < 4.0))
    return float(np.sum(score * pair) / pair.sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_path", type=str, nargs="?", default="artifacts/checkpoints/last.ckpt")
    parser.add_argument("--max-samples", type=int, default=3)
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt_path)
    if not ckpt_path.exists():
        print(f"❌ Checkpoint 不存在: {ckpt_path}"); sys.exit(1)

    print(f"加载 checkpoint: {ckpt_path}")
    model = SimpleFold.load_from_checkpoint(str(ckpt_path), map_location=device, strict=False)
    model.eval().to(device)
    print(f"ESM 模型: {model.hparams.esm_model}\n")

    # 准备数据
    tokenizer = BoltzTokenizer()
    featurizer = BoltzFeaturizer()
    target_dir = Path("data/processed_targets")

    aa_map = {0:"X",1:"A",2:"R",3:"N",4:"D",5:"C",6:"Q",7:"E",8:"G",9:"H",
              10:"I",11:"L",12:"K",13:"M",14:"F",15:"P",16:"S",17:"T",18:"W",19:"Y",20:"V",21:"X"}

    records = sorted(json.load(open(target_dir / "manifest.json")), key=lambda x: x["id"])[:args.max_samples]
    results = []

    for rec in records:
        pdb_id = rec["id"].lower()
        print(f"{'─'*42}")
        print(f"{pdb_id}: ", end="", flush=True)

        record = Record.from_dict(rec)
        valid_chains = [c for c in record.chains if c.valid]
        record = replace(record, chains=valid_chains)
        if not record.chains:
            print("跳过 (无有效链)"); continue

        input_data = load_input(record, target_dir)
        tokenized = tokenizer.tokenize(input_data)
        seq = "".join(aa_map.get(t["res_type"], "X") for t in tokenized.tokens)
        n_tokens = len(tokenized.tokens)
        print(f"{n_tokens} 残基, ", end="", flush=True)

        # 特征化（不 pad）
        raw_features = featurizer.process(
            tokenized, max_atoms=None, max_tokens=None,
            symmetries={}, atoms_per_window_queries=32,
            min_dist=2.0, max_dist=22.0, num_bins=64,
            compute_symmetries=False, rotation_augment_coords=False,
        )

        # 转 tensor
        tensor_features = {}
        for k, v in raw_features.items():
            if isinstance(v, np.ndarray):
                tensor_features[k] = torch.from_numpy(v)
            else:
                tensor_features[k] = v

        # coords 有 featurizer 加的 batch 维 [1, N, 3] -> [N, 3]
        tensor_features["coords"] = tensor_features["coords"].squeeze(0)

        # 添加用户字段（用标量 tensor 避免 collate 多 stack 出多余维度）
        tensor_features["aa_seq"] = seq
        tensor_features["max_num_tokens"] = torch.tensor(n_tokens)  # scalar
        tensor_features["cropped_num_tokens"] = torch.tensor(n_tokens)  # scalar

        # 用训练代码的 collate 加 batch 维度
        batch = collate([tensor_features])
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # 补充模型需要的但 featurizer 没有的字段
        batch["atom_to_token_idx"] = batch["atom_to_token"].argmax(dim=-1)  # [B, N]

        # ESM 特征
        processor = ProteinDataProcessor(device, multiplicity=1)
        with torch.no_grad():
            processor.process_esm(batch, model.esm_model, model.esm_dict, model.af2_to_esm, inference=True)

        # 前向 (t=0.5)
        t = torch.full((1,), 0.5, device=device)
        noise = torch.randn_like(batch["coords"])
        noised = batch["coords"] * (1 - t.view(-1, 1, 1)) + noise * t.view(-1, 1, 1)

        with torch.no_grad():
            out = model.model(noised, t, batch)
            pred_coords = noised + out["predict_velocity"] * (1 - t.view(-1, 1, 1))

        # 评估
        mask = batch["atom_pad_mask"].bool()
        true_coords = batch["coords"]
        scale = 16.0
        rmsd = compute_rmsd(pred_coords[0] * scale, true_coords[0] * scale, mask[0])
        lddt = compute_lddt(pred_coords[0] * scale, true_coords[0] * scale, mask[0])
        results.append((pdb_id, n_tokens, rmsd, lddt))
        print(f"RMSD={rmsd:6.2f}Å  lDDT={lddt:.3f}")

    # 汇总
    if results:
        avg_r, avg_l = np.mean([r[2] for r in results]), np.mean([r[3] for r in results])
        print(f"\n{'═'*42}")
        print(f"平均: RMSD={avg_r:6.2f}Å  lDDT={avg_l:.3f}")
        print(f"\n参考:")
        print(f"  随机猜测     → RMSD~10-20Å  lDDT~0.25")
        print(f"  可用水平     → RMSD~3-5Å    lDDT~0.60")
        print(f"  AlphaFold2   → RMSD~1-3Å    lDDT~0.80")
        if avg_r < 8:
            print("✅ 模型有初步学习效果")
        elif avg_r < 15:
            print("🔄 开始收敛，但还需要很多步训练")
        else:
            print("ℹ️  接近随机状态 (5步CPU训练正常)")
    else:
        print("⚠ 没有成功样本")


if __name__ == "__main__":
    main()
