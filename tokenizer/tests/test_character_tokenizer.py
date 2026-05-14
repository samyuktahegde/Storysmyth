"""
Tests for Character Tokenization
"""

import pytest
import json
from tokenizer import CharacterTokenizer

def test_character_tokenizer_can_be_imported():
    from tokenizer import CharacterTokenizer
    assert CharacterTokenizer is not None

def test_vocabulary_is_sorted():
    tokenizer = CharacterTokenizer()
    tokenizer.fit(["Loia", "Antaka", "Unbardak","Bitobi"])
    vocabulary = tokenizer.get_vocabulary()
    # characters = list(vocabulary.keys())
    special_tokens = {"<PAD>", "<BOS>", "<EOS>"}
    characters = [ch for ch in vocabulary.keys() if ch not in special_tokens]
    assert characters == sorted(characters)

def test_sorted_vocabulary_id_order():
    tokenizer = CharacterTokenizer()
    tokenizer.fit(["Loia", "Antaka", "Unbardak","Bitobi"])
    vocabulary = tokenizer.get_vocabulary()
    assert min(vocabulary.values()) == 0

def test_encode_token_to_id():
    tokenizer = CharacterTokenizer()
    tokenizer.fit(["Loia", "Antaka", "Unbardak","Bitobi"])
    assert tokenizer.encode("Loia") == [5, 13, 10, 7]

def test_decode_id_to_token():
    tokenizer = CharacterTokenizer()
    tokenizer.fit(["Loia", "Antaka", "Unbardak","Bitobi"])
    assert tokenizer.decode(tokenizer.encode("Loia")) == "Loia"

def test_encode_raises_key_error_for_unknown_characters():
    tokenizer = CharacterTokenizer()
    tokenizer.fit(["Loia", "Antaka", "Unbardak","Bitobi"])
    with pytest.raises(KeyError):
        tokenizer.encode("Zzz")

def test_encode_before_fit_raises_runtime_error():
    tokenizer = CharacterTokenizer()
    with pytest.raises(RuntimeError):
        tokenizer.encode("Loia")

def test_vocabulary_size():
    tokenizer = CharacterTokenizer()
    tokenizer.fit(["Loia", "Antaka", "Unbardak","Bitobi"])
    assert len(tokenizer.get_vocabulary()) == 16

def test_vocabulary_size_raises_runtime_error():
    tokenizer = CharacterTokenizer()
    with pytest.raises(RuntimeError):
       _ = tokenizer.vocabulary_size

def test_save(tmp_path):
    tokenizer = CharacterTokenizer()
    tokenizer.fit(["Loia", "Antaka", "Unbardak","Bitobi"])
    save_path = tmp_path / "test_tokenizer.json"
    tokenizer.save(str(save_path))
    assert save_path.exists()
    with open(save_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    assert "token_to_id" in payload
    assert payload["token_to_id"] == {"<BOS>": 1, "<EOS>": 2, "<PAD>": 0, "A": 3, "B": 4, "L": 5, "U": 6, "a": 7, "b": 8, "d": 9, "i": 10, "k": 11, "n": 12, "o": 13, "r": 14, "t": 15}

def test_load(tmp_path):
    tokenizer = CharacterTokenizer()
    tokenizer.fit(["Loia", "Antaka", "Unbardak","Bitobi"])
    save_path = tmp_path / "test_tokenizer.json"
    tokenizer.save(str(save_path))
    loaded = CharacterTokenizer.load(str(save_path))
    assert loaded.encode("Loia") == [5, 13, 10, 7]
    assert loaded.decode([5, 13, 10, 7]) == "Loia"
    assert loaded._fitted == True

def test_bos_token_id_is_1():
    assert CharacterTokenizer.BOS == 1

def test_eos_token_id_is_1():
    assert CharacterTokenizer.EOS == 2

def test_pad_token_id_is_1():
    assert CharacterTokenizer.PAD == 0

def test_corpus_characters_ids_start_from_3():
    tokenizer = CharacterTokenizer()
    tokenizer.fit(["Loia", "Antaka", "Unbardak","Bitobi"])
    assert(min(tokenizer.get_vocabulary(include_special=False).values()) == 3)

def test_special_tokens_are_reserved():
    tokenizer = CharacterTokenizer()
    tokenizer.fit(["Loia", "Antaka", "Unbardak","Bitobi"])
    vocabulary = tokenizer.get_vocabulary()
    assert vocabulary["<PAD>"] == 0
    assert vocabulary["<BOS>"] == 1
    assert vocabulary["<EOS>"] == 2

