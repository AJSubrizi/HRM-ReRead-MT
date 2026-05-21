"""Laptop-sized reference training loop for re-reading with latent memory.

This is not the full 1B-parameter HRM training recipe. It is an executable
reference that demonstrates the loss plumbing and memory-buffer mechanics on
small tokenized text samples.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class TinyHierarchicalReader(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.low = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.high = nn.GRUCell(hidden_size, hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(
        self, input_ids: torch.Tensor, prior_high: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        embeddings = self.embed(input_ids)
        low_states, _ = self.low(embeddings)
        pooled = low_states[:, -1]
        if prior_high is None:
            prior_high = torch.zeros_like(pooled)
        high_state = self.high(pooled, prior_high)
        logits = self.head(high_state)
        return logits, high_state


def build_char_vocab(texts: List[str]) -> Dict[str, int]:
    chars = sorted({char for text in texts for char in text})
    return {"<pad>": 0, **{char: index + 1 for index, char in enumerate(chars)}}


def encode(text: str, vocab: Dict[str, int], max_length: int) -> List[int]:
    ids = [vocab[char] for char in text[:max_length]]
    return ids + [0] * (max_length - len(ids))


def load_texts(path: Path) -> List[str]:
    texts: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            source = row.get("teacher_response") or row.get("answer") or row.get("prompt")
            if source:
                texts.append(str(source))
    if not texts:
        raise ValueError(f"No usable text rows found in {path}")
    return texts


def train_reference(
    data_path: Path,
    *,
    epochs: int = 5,
    max_length: int = 128,
    hidden_size: int = 128,
    lr: float = 1e-3,
) -> List[float]:
    texts = load_texts(data_path)
    vocab = build_char_vocab(texts)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyHierarchicalReader(len(vocab), hidden_size=hidden_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    inputs = torch.tensor([encode(text, vocab, max_length) for text in texts], device=device)
    targets = inputs[:, 0].clone()
    latent_memory: Optional[torch.Tensor] = None
    losses: List[float] = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        prior = latent_memory.detach() if latent_memory is not None else None
        logits, high_state = model(inputs, prior)
        pred_loss = F.cross_entropy(logits, targets)
        cons_loss = torch.tensor(0.0, device=device)
        if prior is not None:
            cons_loss = F.mse_loss(high_state, prior)
        loss = pred_loss + 0.1 * cons_loss
        loss.backward()
        optimizer.step()
        latent_memory = high_state.detach()
        losses.append(float(loss.detach().cpu()))
        print(f"epoch={epoch + 1} loss={losses[-1]:.4f}")
    return losses


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="JSONL data path.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=128)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train_reference(
        args.data,
        epochs=args.epochs,
        max_length=args.max_length,
        hidden_size=args.hidden_size,
    )


if __name__ == "__main__":
    main()
