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
    characters = list(vocabulary.keys())
    assert characters == sorted(characters)

def test_sorted_vocabulary_id_order():
    tokenizer = CharacterTokenizer()
    tokenizer.fit(["Loia", "Antaka", "Unbardak","Bitobi"])
    vocabulary = tokenizer.get_vocabulary()
    assert min(vocabulary.values()) == 0

def test_encode_token_to_id():
    tokenizer = CharacterTokenizer()
    tokenizer.fit(["Loia", "Antaka", "Unbardak","Bitobi"])
    assert tokenizer.encode("Loia") == [2, 10, 7, 4]

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
    assert len(tokenizer.get_vocabulary()) == 13

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
    assert payload["token_to_id"] == {"A": 0, "B": 1, "L": 2, "U": 3, "a": 4, "b": 5, "d": 6, "i": 7, "k": 8, "n": 9, "o": 10, "r": 11, "t": 12}

def test_load(tmp_path):
    tokenizer = CharacterTokenizer()
    tokenizer.fit(["Loia", "Antaka", "Unbardak","Bitobi"])
    save_path = tmp_path / "test_tokenizer.json"
    tokenizer.save(str(save_path))
    loaded = CharacterTokenizer.load(str(save_path))
    assert loaded.encode("Loia") == [2, 10, 7, 4]
    assert loaded.decode([2, 10, 7, 4]) == "Loia"
    assert loaded._fitted == True
