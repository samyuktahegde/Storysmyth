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

def test_embedding_dimensions():
    model = NameLSTM(vocab_size=50, embed_dim=32, hidden_dim=64)
    assert model.embedding.num_embeddings == 50
    assert model.embedding.embedding_dim == 32

def test_lstm_dimensions():
    model = NameLSTM(vocab_size=50, embed_dim=32, hidden_dim=64)
    assert model.lstm.input_size == 32
    assert model.lstm.hidden_size == 64

def test_linear_dimensions():
    model = NameLSTM(vocab_size=50, embed_dim=32, hidden_dim=64)
    assert model.linear.in_features == 64
    assert model.linear.out_features == 50

def test_num_layers():
    model = NameLSTM(vocab_size=50, embed_dim=32, hidden_dim=64, num_layers=2)
    assert model.lstm.num_layers == 2