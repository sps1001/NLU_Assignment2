"""
Dataset utilities shared across all three models.
CSL 7640 - NLU Assignment 2, Problem 2
Author: B23CS1061

Handles:
  - Loading and cleaning the name list
  - Building a character-level vocabulary
  - Encoding names as integer sequences with <SOS> and <EOS> tokens
  - PyTorch Dataset + DataLoader construction
"""

import torch
from torch.utils.data import Dataset, DataLoader

# Special tokens
PAD_TOKEN = 0   # padding (not used in training, reserved)
SOS_TOKEN = 1   # start-of-sequence
EOS_TOKEN = 2   # end-of-sequence


def load_names(path: str) -> list[str]:
    """
    Read one name per line, strip whitespace, lowercase, keep non-empty.
    Returns a sorted, deduplicated list for reproducibility.
    """
    with open(path, encoding="utf-8") as f:
        names = [line.strip().lower() for line in f if line.strip()]
    # Deduplicate while preserving rough order (use dict to keep insertion order)
    seen = {}
    for n in names:
        seen[n] = True
    return list(seen.keys())


def build_vocab(names: list[str]) -> tuple[dict, dict]:
    """
    Build char→index and index→char mappings.
    Index 0 = PAD, 1 = SOS, 2 = EOS, 3..N = alphabet characters.
    """
    chars = sorted(set(ch for name in names for ch in name))
    char2idx = {ch: i + 3 for i, ch in enumerate(chars)}
    char2idx["<PAD>"] = PAD_TOKEN
    char2idx["<SOS>"] = SOS_TOKEN
    char2idx["<EOS>"] = EOS_TOKEN
    idx2char = {v: k for k, v in char2idx.items()}
    return char2idx, idx2char


def encode_name(name: str, char2idx: dict) -> list[int]:
    """
    Encode a name string as [SOS, c1, c2, ..., cn, EOS].
    Unknown characters are skipped.
    """
    tokens = [SOS_TOKEN]
    for ch in name:
        if ch in char2idx:
            tokens.append(char2idx[ch])
    tokens.append(EOS_TOKEN)
    return tokens


class NamesDataset(Dataset):
    """
    PyTorch Dataset for character-level name sequences.
    Each item is a 1-D LongTensor of encoded character indices
    including SOS and EOS tokens.
    """

    def __init__(self, names: list[str], char2idx: dict):
        # Encode every name once; store as tensors for fast indexing
        self.data = [
            torch.tensor(encode_name(n, char2idx), dtype=torch.long)
            for n in names
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pad a batch of variable-length sequences to the same length.
    Returns:
        padded  – (batch, max_len) LongTensor
        lengths – (batch,) LongTensor with original lengths
    """
    lengths = torch.tensor([seq.size(0) for seq in batch], dtype=torch.long)
    padded = torch.nn.utils.rnn.pad_sequence(
        batch, batch_first=True, padding_value=PAD_TOKEN
    )
    return padded, lengths


def get_dataloader(
    names: list[str],
    char2idx: dict,
    batch_size: int = 64,
    shuffle: bool = True,
) -> DataLoader:
    """Create a DataLoader from a list of name strings."""
    ds = NamesDataset(names, char2idx)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
