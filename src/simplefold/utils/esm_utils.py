#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

# Started from https://github.com/facebookresearch/esm/tree/main,
# licensed under MIT License, Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
import typing as T
import numpy as np
from functools import partial

from utils import residue_constants

try:
    import mlx.core as mx
except:
    pass

# Try torch.hub first, fall back to transformers if that fails
def _load_esm_hub(model_name: str):
    """Load ESM model, falling back to transformers if torch.hub fails."""
    try:
        return torch.hub.load(
            "facebookresearch/esm:main", model_name, trust_repo=True
        )
    except Exception:
        pass

    # Fallback: load via HuggingFace transformers
    try:
        from transformers import EsmModel, EsmTokenizer

        # Map ESM hub model names to HuggingFace IDs
        hf_map = {
            "esm2_t6_8M_UR50D": "facebook/esm2_t6_8M_UR50D",
            "esm2_t12_35M_UR50D": "facebook/esm2_t12_35M_UR50D",
            "esm2_t30_150M_UR50D": "facebook/esm2_t30_150M_UR50D",
            "esm2_t33_650M_UR50D": "facebook/esm2_t33_650M_UR50D",
            "esm2_t36_3B_UR50D": "facebook/esm2_t36_3B_UR50D",
            "esm2_t48_15B_UR50D": "facebook/esm2_t48_15B_UR50D",
        }
        hf_id = hf_map.get(model_name)
        if hf_id is None:
            raise ValueError(f"Unknown ESM model: {model_name}")

        hf_model = EsmModel.from_pretrained(hf_id)
        hf_tokenizer = EsmTokenizer.from_pretrained(hf_id)

        # Create wrapper to match ESM hub model API
        class ESMHubModelWrapper:
            """Wraps transformers EsmModel to expose hub-style call interface."""
            def __init__(self, model):
                self._model = model
                self.num_layers = model.config.num_hidden_layers

            def __call__(self, esmaa, repr_layers=None, need_head_weights=False):
                with torch.no_grad():
                    outputs = self._model(
                        input_ids=esmaa,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    hidden = outputs.hidden_states
                    reps = {}
                    if repr_layers is not None:
                        for i in repr_layers:
                            # hidden_states[0] = embedding, [1] = layer 0, [2] = layer 1, ...
                            reps[i] = hidden[i]
                    return {"representations": reps}

            def to(self, device):
                self._model = self._model.to(device)
                return self

            def eval(self):
                self._model = self._model.eval()
                return self

            def __getattr__(self, name):
                return getattr(self._model, name)

        from types import SimpleNamespace
        alphabet = SimpleNamespace(
            cls_idx=hf_tokenizer.cls_token_id,
            eos_idx=hf_tokenizer.eos_token_id,
            padding_idx=hf_tokenizer.pad_token_id,
            vocab=hf_tokenizer.get_vocab(),
        )
        alphabet.get_idx = lambda token: alphabet.vocab.get(token, alphabet.padding_idx)

        return ESMHubModelWrapper(hf_model), alphabet
    except ImportError:
        raise ImportError(
            "Could not load ESM model via torch.hub or transformers. "
            "Install with: pip install transformers"
        )
    except ImportError:
        raise ImportError(
            "Could not load ESM model via torch.hub (rate limited) or transformers (not installed). "
            "Install transformers: pip install transformers"
        )


def _make_loader(model_name):
    return lambda: _load_esm_hub(model_name)


esm_registry = {
    "esm2_8M": _make_loader("esm2_t6_8M_UR50D"),
    "esm2_35M": _make_loader("esm2_t12_35M_UR50D"),
    "esm2_150M": _make_loader("esm2_t30_150M_UR50D"),
    "esm2_650M": _make_loader("esm2_t33_650M_UR50D"),
    "esm2_3B": _make_loader("esm2_t36_3B_UR50D"),
    "esm2_15B": _make_loader("esm2_t48_15B_UR50D"),
}


esm_model_dict = {
    "esm2_8M": {
        "esm_s_dim": 320,
        "esm_z_dim": 120,
        "esm_num_layers": 7,
    },
    "esm2_35M": {
        "esm_s_dim": 480,
        "esm_z_dim": 240,
        "esm_num_layers": 13,
    },
    "esm2_150M": {
        "esm_s_dim": 640,
        "esm_z_dim": 600,
        "esm_num_layers": 31,
    },
    "esm2_650M": {
        "esm_s_dim": 1280,
        "esm_z_dim": 660,
        "esm_num_layers": 34,
    },
    "esm2_3B": {
        "esm_s_dim": 2560,
        "esm_z_dim": 1440,
        "esm_num_layers": 37,
    },
    "esm2_15B": {
        "esm_s_dim": 5120,
        "esm_z_dim": 1920,
        "esm_num_layers": 49,
    },
}


def encode_sequence(
    seq: str,
    residue_index_offset: T.Optional[int] = 512,
    chain_linker: T.Optional[str] = "G" * 25,
) -> T.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if chain_linker is None:
        chain_linker = ""
    if residue_index_offset is None:
        residue_index_offset = 0

    chains = seq.split(":")
    seq = chain_linker.join(chains)

    unk_idx = residue_constants.restype_order_with_x["X"]
    encoded = torch.tensor(
        [residue_constants.restype_order_with_x.get(aa, unk_idx) for aa in seq]
    )
    residx = torch.arange(len(encoded))

    if residue_index_offset > 0:
        start = 0
        for i, chain in enumerate(chains):
            residx[start : start + len(chain) + len(chain_linker)] += (
                i * residue_index_offset
            )
            start += len(chain) + len(chain_linker)

    linker_mask = torch.ones_like(encoded, dtype=torch.float32)
    chain_index = []
    offset = 0
    for i, chain in enumerate(chains):
        if i > 0:
            chain_index.extend([i - 1] * len(chain_linker))
        chain_index.extend([i] * len(chain))
        offset += len(chain)
        linker_mask[offset : offset + len(chain_linker)] = 0
        offset += len(chain_linker)

    chain_index = torch.tensor(chain_index, dtype=torch.int64)

    return encoded, residx, linker_mask, chain_index


def batch_encode_sequences(
    sequences: T.Sequence[str],
    residue_index_offset: T.Optional[int] = 512,
    chain_linker: T.Optional[str] = "G" * 25,
) -> T.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    aatype_list = []
    residx_list = []
    linker_mask_list = []
    chain_index_list = []
    for seq in sequences:
        aatype_seq, residx_seq, linker_mask_seq, chain_index_seq = encode_sequence(
            seq,
            residue_index_offset=residue_index_offset,
            chain_linker=chain_linker,
        )
        aatype_list.append(aatype_seq)
        residx_list.append(residx_seq)
        linker_mask_list.append(linker_mask_seq)
        chain_index_list.append(chain_index_seq)

    aatype = collate_dense_tensors(aatype_list)
    mask = collate_dense_tensors(
        [aatype.new_ones(len(aatype_seq)) for aatype_seq in aatype_list]
    )
    residx = collate_dense_tensors(residx_list)
    linker_mask = collate_dense_tensors(linker_mask_list)
    chain_index_list = collate_dense_tensors(chain_index_list, -1)

    return aatype, mask, residx, linker_mask, chain_index_list


def collate_dense_tensors(
    samples: T.List[torch.Tensor], pad_v: float = 0
) -> torch.Tensor:
    """
    Takes a list of tensors with the following dimensions:
        [(d_11,       ...,           d_1K),
         (d_21,       ...,           d_2K),
         ...,
         (d_N1,       ...,           d_NK)]
    and stack + pads them into a single tensor of:
    (N, max_i=1,N { d_i1 }, ..., max_i=1,N {diK})
    """
    if len(samples) == 0:
        return torch.Tensor()
    if len(set(x.dim() for x in samples)) != 1:
        raise RuntimeError(
            f"Samples has varying dimensions: {[x.dim() for x in samples]}"
        )
    (device,) = tuple(set(x.device for x in samples))  # assumes all on same device
    max_shape = [max(lst) for lst in zip(*[x.shape for x in samples])]
    result = torch.empty(
        len(samples), *max_shape, dtype=samples[0].dtype, device=device
    )
    result.fill_(pad_v)
    for i in range(len(samples)):
        result_i = result[i]
        t = samples[i]
        result_i[tuple(slice(0, k) for k in t.shape)] = t
    return result


def _af2_to_esm(d):
    # Remember that t is shifted from residue_constants by 1 (0 is padding).
    esm_reorder = [d.padding_idx] + [
        d.get_idx(v) for v in residue_constants.restypes_with_x
    ]
    return torch.tensor(esm_reorder)


def af2_idx_to_esm_idx(aa, mask, af2_to_esm):
    aa = (aa + 1).masked_fill(mask != 1, 0)
    return af2_to_esm[aa]


def compute_language_model_representations(
    esmaa, esm, esm_dict, backend="torch"
) -> torch.Tensor:
    """Adds bos/eos tokens for the language model, since the structure module doesn't use these."""
    batch_size = esmaa.size(0)

    bosi, eosi = esm_dict.cls_idx, esm_dict.eos_idx
    bos = esmaa.new_full((batch_size, 1), bosi)
    eos = esmaa.new_full((batch_size, 1), esm_dict.padding_idx)
    esmaa = torch.cat([bos, esmaa, eos], dim=1)
    # Use the first padding index as eos during inference.
    esmaa[range(batch_size), (esmaa != 1).sum(1)] = eosi

    if backend == "mlx":
        esmaa = mx.array(esmaa)

    res = esm(
        esmaa,
        repr_layers=range(esm.num_layers + 1),
        need_head_weights=False,
    )
    if backend == "mlx":
        res['representations'] = {k: torch.from_numpy(np.array(v)) for k,v in res['representations'].items()}

    esm_s = torch.stack(
        [v for _, v in sorted(res["representations"].items())], dim=2
    )
    esm_s = esm_s[:, 1:-1]  # B, L, nLayers, C
    return esm_s, None
