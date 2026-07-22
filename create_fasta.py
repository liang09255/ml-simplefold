#!/usr/bin/env python3
"""
从 data/mmcif_raw/*.cif 提取蛋白质序列，生成 FASTA 文件。
"""
import gemmi
from pathlib import Path

mmcif_dir = Path("data/mmcif_raw")
fasta_dir = Path("data/fasta")
fasta_dir.mkdir(exist_ok=True)

for cif_file in sorted(mmcif_dir.glob("*.cif")):
    doc = gemmi.cif.read(str(cif_file))
    block = doc.sole_block()

    # 尝试从 _entity_poly 中获取序列
    seq = None

    # 方法1: pdbx_seq_one_letter_code
    try:
        seq = block.find("_entity_poly.", ["pdbx_seq_one_letter_code"])[0].str(0)
        seq = seq.replace("\n", "").replace(" ", "")
    except (IndexError, RuntimeError):
        pass

    # 方法2: 从 _entity_poly_seq 拼接
    if seq is None:
        try:
            tags = block.find_values("_entity_poly_seq.mon_id")
            # 用 residue_constants 转换三字母到单字母
            letters = []
            for tag in tags:
                three_letter = tag.str(0).strip()
                # 基本映射（只做标准残基）
                aa_map = {
                    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
                    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
                    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
                    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
                }
                letters.append(aa_map.get(three_letter, "X"))
            seq = "".join(letters)
        except (IndexError, RuntimeError):
            pass

    if seq is None:
        print(f"⚠  {cif_file.stem}: 无法提取序列，跳过")
        continue

    pdb_id = cif_file.stem.lower()
    with open(fasta_dir / f"{pdb_id}.fasta", "w") as f:
        f.write(f">{pdb_id.upper()}\n{seq}\n")
    print(f"✓  {pdb_id}: {len(seq)} 个残基 -> data/fasta/{pdb_id}.fasta")
