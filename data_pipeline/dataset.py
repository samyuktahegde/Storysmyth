import csv
from curses import raw
import warnings
import torch

from tokenizer.character_tokenizer import CharacterTokenizer

class NameDataset:

    def __init__(self, source, tokenizer):
        self._tokenizer = tokenizer
        self._sequences = []
        if isinstance(source, str):
            file = open(source, newline="", encoding="utf-8")
            close_after = True
        else:
            file = source
            close_after = False
        try:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                raw = row[0]
                parts = raw.split(":", 1)
                if len(parts) != 2:
                    warnings.warn(f"Skipping malformed row: {raw!r}")
                    continue
                race, name = parts
                race = race.strip()
                name = name.strip()
                if not race or not name:
                    warnings.warn(f"Malformed row skipped: {raw!r}")
                    continue
                encoded = [CharacterTokenizer.BOS] + tokenizer.encode(f"{race}:{name}") + [CharacterTokenizer.EOS]
                self._sequences.append(encoded)
        finally:
            if close_after:
                file.close()
        if not self._sequences:
            raise ValueError("No valid rows found in that dataset.")
        max_length = max(len(s) for s in self._sequences)
        self._sequences = [s + [CharacterTokenizer.PAD] * (max_length - len(s)) for s in self._sequences]

    def __getitem__(self, index):
        if index < 0 or index >= len(self._sequences):
            raise IndexError(f"Index {index} is out of range for dataset of size {len(self._sequences)}.")
        return torch.tensor(self._sequences[index])

    def __len__(self):
        return len(self._sequences)
