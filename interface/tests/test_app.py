import pytest
from interface import app
from tokenizer.character_tokenizer import CharacterTokenizer

def test_parser_requires_race():
    parser = app.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_parser_defaults():
    args = app.build_parser().parse_args(["--race", "elf"])
    assert args.race == "elf"
    assert args.run == "checkpoints/run1"
    assert args.epoch == 50
    assert args.temperature == 1.0

def test_parser_overrides():
    args = app.build_parser().parse_args(["--race", "orc", "--run", "checkpoints/run2", "--epoch", "10", "--temperature", "0.7"])
    assert args.race == "orc"
    assert args.run == "checkpoints/run2"
    assert args.epoch == 10
    assert args.temperature == 0.7

def test_checkpoint_path_pads_epoch():
    assert app.get_checkpoint_path("checkpoints/run1", 5) == "checkpoints/run1/epoch_05.pt"
    assert app.get_checkpoint_path("checkpoints/run1", 50) == "checkpoints/run1/epoch_50.pt"

def test_load_tokenizer_reads_run_folder(tmp_path):
    t = CharacterTokenizer()
    t.fit(["elf:Loia, elf"])
    t.save(str(tmp_path / "tokenizer.json"))

    loaded = app.load_tokenizer(str(tmp_path))
    assert loaded.get_vocabulary() == t.get_vocabulary()

def test_main_prints_one_generated_name(monkeypatch, capsys, tmp_path):
    t = CharacterTokenizer()
    t.fit(["elf:Loia, elf"])
    t.save(str(tmp_path / "tokenizer.json"))
    calls = {}

    def fake_load_model(path, vocab_size, **kwargs):
        calls["checkpoint"] = path
        calls["vocab_size"] = vocab_size
        return "MODEL"
    
    def fake_generate(model, tokenizer, prompt, temperature=1.0, max_length=50):
        calls["model"] = model
        calls["prompt"] = prompt
        calls["temperature"] = temperature
        return "Iaea"

    monkeypatch.setattr(app, "load_model", fake_load_model)
    monkeypatch.setattr(app, "generate", fake_generate)

    app.main(["--race", "elf", "--run", str(tmp_path), "--epoch", "7", "--temperature", "0.8"])

    assert capsys.readouterr().out.strip() == "Iaea"
    assert calls["checkpoint"] == str(tmp_path / "epoch_07.pt")
    assert calls["vocab_size"] == t.vocabulary_size
    assert calls["model"] == "MODEL"
    assert calls["prompt"] == "elf:"
    assert calls["temperature"] == 0.8
