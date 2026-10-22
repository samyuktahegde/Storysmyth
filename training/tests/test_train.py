import os
import torch
import tempfile
from model.lstm.lstm import NameLSTM
from training.train import save_checkpoint, build_optimizer, build_criterion
from tokenizer.character_tokenizer import CharacterTokenizer
from training import train as train_module

def test_save_checkpoint_creates_file():
    model = NameLSTM(vocab_size=50, embed_dim=32, hidden_dim=64)
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = save_checkpoint(model, epoch=1, checkpoint_dir=tmp_dir)
        assert os.path.exists(path)

def test_save_checkpoint_filename():
    model = NameLSTM(vocab_size=50, embed_dim=32, hidden_dim=64)
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = save_checkpoint(model, epoch=5, checkpoint_dir=tmp_dir)
        assert os.path.basename(path) == "epoch_05.pt"

def test_build_optimizer_returns_adam():
    model = NameLSTM(vocab_size=50, embed_dim=32, hidden_dim=64)
    optimizer = build_optimizer(model)
    assert isinstance(optimizer, torch.optim.Adam)

def test_build_criterion_returns_cross_entropy():
    criterion = build_criterion()
    assert isinstance(criterion, torch.nn.CrossEntropyLoss)

def test_train_saves_tokenizer_and_checkpoints_in_run_dir(tmp_path):
    csv = tmp_path /"fantasy_names.csv"
    rows = ["name, race"]+[f"elf:Loia{i}, elf" for i in range(10)]
    csv.write_text("\n".join(rows)+"\n")
    tokenizer = CharacterTokenizer()
    tokenizer.fit(csv.read_text().splitlines()[1:])
    run_dir = tmp_path / "run"

    train_module.train(str(csv), tokenizer, str(run_dir), epochs=1, batch_size=4)

    assert (run_dir / "tokenizer.json").exists()
    assert(run_dir / "epoch_01.pt").exists()
