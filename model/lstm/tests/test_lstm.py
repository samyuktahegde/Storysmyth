import torch
from model.lstm.lstm import NameLSTM

def test_model_instantiation():
    model = NameLSTM(vocab_size=50, embed_dim=32, hidden_dim=64)
    assert isinstance(model, NameLSTM)

def test_output_shape():
    model = NameLSTM(vocab_size=50, embed_dim=32, hidden_dim=64)
    x = torch.randint(0, 50, (4, 10))
    output = model(x)
    assert output.shape == (4, 10, 50)