import torch
import tempfile
from model.lstm.lstm import NameLSTM
from tokenizer.character_tokenizer import CharacterTokenizer
from generator.generator import encode_prompt, generate, load_model
from training.train import save_checkpoint

def test_encode_prompt_starts_with_bos():
    tokenizer = CharacterTokenizer()
    tokenizer.fit(["elf:Aerin"])
    input_ids = encode_prompt("elf:", tokenizer)
    bos_id = tokenizer._token_to_id["<BOS>"]
    assert input_ids[0,0].item() == bos_id

def test_encode_prompt_shape():
    tokenizer = CharacterTokenizer()
    tokenizer.fit(["elf:Aerin"])
    input_ids = encode_prompt("elf:", tokenizer)
    assert input_ids.shape == (1, 5)

def test_generate_return_string():
    tokenizer = CharacterTokenizer()
    tokenizer.fit(["elf:Aerin"])
    model = NameLSTM(vocab_size=tokenizer.vocabulary_size, embed_dim=32, hidden_dim=64)
    result = generate(model, tokenizer, prompt = "elf:")
    assert isinstance(result, str)

def test_generate_no_special_tokens():
    tokenizer = CharacterTokenizer()
    tokenizer.fit(["elf:Aerin"])
    model = NameLSTM(vocab_size=tokenizer.vocabulary_size, embed_dim=32, hidden_dim=64)
    result = generate(model, tokenizer, prompt = "elf:")
    assert "<BOS>" not in result
    assert "<EOS>" not in result
    assert "<PAD>" not in result

def test_load_model_returns_namelstm():
    tokenizer = CharacterTokenizer()
    tokenizer.fit(["elf:Aerin"])
    model = NameLSTM(vocab_size=tokenizer.vocabulary_size, embed_dim=32, hidden_dim=64)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = save_checkpoint(model, epoch = 1, checkpoint_dir = tmp_dir)
        loaded_model = load_model(path, vocab_size=tokenizer.vocabulary_size, embed_dim=32, hidden_dim=64)
        assert isinstance(loaded_model, NameLSTM)
