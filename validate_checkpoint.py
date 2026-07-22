#!/usr/bin/env python3
"""
validate_checkpoint.py — 用已有测试数据验证 checkpoint 是否正常工作

流程:
  1. 加载 checkpoint 模型
  2. 从 data/processed_targets 加载样本
  3. 用 processor 提取 ESM 特征
  4. 跑一次前向传播
  5. 计算预测坐标与真实坐标的 RMSD 和 lDDT

用法:
  python validate_checkpoint.py <checkpoint_path>
  python validate_checkpoint.py artifacts/checkpoints/last.ckpt
  python validate_checkpoint.py artifacts/checkpoints/model-best-step00000005-loss0.000000.ckpt
"""
import os
import sys
import pickle
import json
import argparse
import warnings
import numpy as np
from pathlib import Path

# 忽略烦人的 warning
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

sys.path.insert(0, "src/simplefold")

import torch
import lightning.pytorch as pl
from omegaconf import OmegaConf

from model.simplefold import SimpleFold
from processor.protein_processor import ProteinDataProcessor
from boltz_data_pipeline.tokenize.boltz_protein import BoltzTokenizer
from boltz_data_pipeline.feature.featurizer import BoltzFeaturizer
from boltz_data_pipeline.types import Manifest, Record, Structure, Input
from utils.datamodule_utils import load_input

torch.set_float32_matmul_precision("medium")


def compute_rmsd(pred, true, mask):
    """刚性对齐后计算 RMSD"""
    pred = pred[mask].cpu().numpy()
    true = true[mask].cpu().numpy()

    # 中心化
    pc = pred - pred.mean(axis=0)
    tc = true - true.mean(axis=0)

    # SVD 刚性对齐
    u, _, vh = np.linalg.svd(pc.T @ tc)
    r = u @ vh
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vh
    aligned = pc @ r.T
    diff = aligned - tc
    rmsd = np.sqrt(np.mean(np.sum(diff ** 2, axis=-1)))
    return rmsd


def compute_lddt(pred, true, mask, cutoff=15.0):
    """计算 lDDT 分数 (0~1)"""
    pred = pred[mask].cpu().numpy()
    true = true[mask].cpu().numpy()

    pdm = np.sqrt(np.sum((pred[:, None] - pred[None, :]) ** 2, axis=-1))
    tdm = np.sqrt(np.sum((true[:, None] - true[None, :]) ** 2, axis=-1))

    n = len(pred)
    pair_mask = np.ones((n, n), dtype=bool)
    np.fill_diagonal(pair_mask, False)

    valid = pair_mask & (tdm < cutoff)
    if valid.sum() == 0:
        return 0.0

    diff = np.abs(pdm - tdm)
    score = 0.25 * ((diff < 0.5) + (diff < 1.0) + (diff < 2.0) + (diff < 4.0))
    return float(np.sum(score * valid) / valid.sum())


def main():
    parser = argparse.ArgumentParser(description="验证 SimpleFold checkpoint")
    parser.add_argument("ckpt_path", type=str, help="checkpoint 路径")
    parser.add_argument("--max_samples", type=int, default=3, help="测试样本数")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt_path)
    if not ckpt_path.exists():
        print(f"❌ Checkpoint 不存在: {ckpt_path}")
        sys.exit(1)

    device = torch.device("cpu")
    print(f"设备: {device}")
    print(f"加载 checkpoint: {ckpt_path}")

    # 加载 checkpoint
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # 获取超参数
    hparams = ckpt.get("hyper_parameters", ckpt.get("hparams", {}))
    if not hparams:
        print("⚠  checkpoint 中未找到超参数，使用默认值")
        esm_model_name = "esm2_8M"
    else:
        esm_model_name = hparams.get("esm_model", "esm2_8M")

    print(f"ESM 模型: {esm_model_name}")

    # 加载模型
    model = SimpleFold.load_from_checkpoint(
        str(ckpt_path),
        map_location=device,
        strict=False,
    )
    model.eval()
    model.to(device)

    # 加载 processor
    processor = ProteinDataProcessor(
        device=device, scale=16.0, ref_scale=5.0, multiplicity=1
    )
    tokenizer = BoltzTokenizer()
    featurizer = BoltzFeaturizer()

    # 加载测试数据
    target_dir = Path("data/processed_targets")
    record_files = sorted((target_dir / "records").glob("*.json"))

    print(f"\n共找到 {len(record_files)} 个样本，测试前 {args.max_samples} 个\n")

    results = []
    for rf in record_files[: args.max_samples]:
        pdb_id = rf.stem.lower()
        print(f"{'='*50}")
        print(f"处理: {pdb_id}")

        # 加载 record
        record = json.load(open(rf))
        record = Record(**record)

        # 只保留有效的链
        record.chains = [c for c in record.chains if c.valid]

        try:
            # 加载结构
            input_data = load_input(record, target_dir)
        except Exception as e:
            print(f"  ⚠  load_input 失败: {e}")
            continue

        try:
            # tokenize
            tokenized = tokenizer.tokenize(input_data)
        except Exception as e:
            print(f"  ⚠ tokenizer 失败: {e}")
            continue

        # 提取序列
        seq = ""
        for token in tokenized.tokens:
            aa_map = {
                0: "X", 1: "A", 2: "R", 3: "N", 4: "D", 5: "C", 6: "Q",
                7: "E", 8: "G", 9: "H", 10: "I", 11: "L", 12: "K",
                13: "M", 14: "F", 15: "P", 16: "S", 17: "T", 18: "W",
                19: "Y", 20: "V", 21: "X",
            }
            seq += aa_map.get(token["res_type"], "X")

        # 构造 batch
        batch = featurizer.process(
            tokenized,
            max_atoms=2304,
            max_tokens=256,
            pad_to_max_atoms=True,
            pad_to_max_tokens=True,
            symmetries={},
            atoms_per_window_queries=32,
            min_dist=2.0,
            max_dist=22.0,
            num_bins=64,
            compute_symmetries=False,
            rotation_augment_ref_pos=False,
            rotation_augment_coords=False,
        )

        batch["aa_seq"] = [seq]
        batch["record"] = [record]
        batch["max_num_tokens"] = torch.tensor([len(tokenized.tokens)])
        batch["cropped_num_tokens"] = torch.tensor([len(tokenized.tokens)])

        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
            elif isinstance(v, np.ndarray):
                batch[k] = torch.from_numpy(v).to(device)

        # ESM 特征提取
        try:
            with torch.no_grad():
                batch = processor.process_esm(
                    batch,
                    esm_model=model.esm_model,
                    esm_dict=model.esm_dict,
                    af2_to_esm=model.af2_to_esm,
                    inference=False,
                )
        except Exception as e:
            print(f"  ⚠ ESM 处理失败: {e}")
            import traceback
            traceback.print_exc()
            continue

        # 前向传播
        n_atoms = batch["coords"].shape[1]
        t = torch.full((1,), 0.5).to(device)
        noise = torch.randn_like(batch["coords"])
        noised_coords = batch["coords"] * (1 - t.view(-1, 1, 1)) + noise * t.view(-1, 1, 1)

        try:
            with torch.no_grad():
                out = model.model(noised_coords, t, batch)

                # 计算 velocity 并得到预测坐标
                v_pred = out
                pred_coords = noised_coords + v_pred * (1.0 - t.view(-1, 1, 1))

                true_coords = batch["coords"]
                atom_mask = batch["atom_pad_mask"].bool()

                # 归一化坐标
                from utils.boltz_utils import center_random_augmentation
                pred_coords = center_random_augmentation(
                    pred_coords, atom_mask, augmentation=False, centering=True
                )
                true_coords = center_random_augmentation(
                    true_coords, atom_mask, augmentation=False, centering=True
                )

                pred_coords = pred_coords * processor.scale
                true_coords = true_coords * processor.scale

                # 计算指标
                rmsd = compute_rmsd(pred_coords[0], true_coords[0], atom_mask[0])
                lddt = compute_lddt(pred_coords[0], true_coords[0], atom_mask[0])

                n_tokens = len(tokenized.tokens)
                print(f"  RMSD = {rmsd:.2f} Å")
                print(f"  lDDT = {lddt:.3f}")
                print(f"  残基数 = {n_tokens}, 原子数 = {n_atoms}")

                results.append({
                    "pdb": pdb_id,
                    "rmsd": rmsd,
                    "lddt": lddt,
                    "n_tokens": n_tokens,
                    "n_atoms": n_atoms,
                })

        except Exception as e:
            print(f"  ⚠ 前向传播失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 汇总
    if results:
        print(f"\n{'='*50}")
        print("汇总:")
        print(f"{'PDB':>6}  {'残基':>4}  {'原子':>5}  {'RMSD(Å)':>8}  {'lDDT':>6}")
        print("-" * 35)
        for r in results:
            print(f"{r['pdb']:>6}  {r['n_tokens']:>4}  {r['n_atoms']:>5}  {r['rmsd']:>8.2f}  {r['lddt']:>6.3f}")
        avg_rmsd = np.mean([r["rmsd"] for r in results])
        avg_lddt = np.mean([r["lddt"] for r in results])
        print("-" * 35)
        print(f"{'平均':>6}  {'':>4}  {'':>5}  {avg_rmsd:>8.2f}  {avg_lddt:>6.3f}")

        # 随机猜测的参照
        print(f"\n参考: 随机猜测 ≈ RMSD 10-20Å, lDDT ≈ 0.20-0.35")
        print(f"      AlphaFold2/SimpleFold 训练完成后 ≈ RMSD 1-3Å, lDDT ≈ 0.7-0.9")
    else:
        print("\n⚠ 没有成功完成任何样本的评估")


if __name__ == "__main__":
    main()
