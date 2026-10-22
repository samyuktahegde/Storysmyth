import os
import torch
from torch.utils.data import DataLoader, random_split
from data_pipeline.dataset import NameDataset
from tokenizer.character_tokenizer import CharacterTokenizer
from model.lstm.lstm import NameLSTM

def load_data(file_path, tokenizer, batch_size=32):
    dataset = NameDataset(file_path, tokenizer)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    return train_loader, val_loader

def build_optimizer(model, lr = 0.001):
    return torch.optim.Adam(model.parameters(), lr=lr)

def build_criterion():
    return torch.nn.CrossEntropyLoss()

def train_epoch(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        optimizer.zero_grad()
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        outputs = model(inputs)
        loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def validate_epoch(model, val_loader, criterion):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch[:,:-1]
            targets = batch[:,1:]
            outputs = model(inputs)
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
            total_loss += loss.item()
    return total_loss / len(val_loader)

def save_checkpoint(model, epoch, checkpoint_dir="checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"epoch_{epoch:02d}.pt")
    torch.save(model.state_dict(), path)
    return path

def train(file_path, tokenizer, run_dir, epochs=50, batch_size=32, lr=0.001):
    os.makedirs(run_dir, exist_ok=True)
    tokenizer.save(os.path.join(run_dir, "tokenizer.json"))
    train_loader, val_loader = load_data(file_path, tokenizer, batch_size)
    vocab_size = tokenizer.vocabulary_size
    model = NameLSTM(vocab_size=vocab_size, embed_dim=32, hidden_dim=64)
    optimizer = build_optimizer(model, lr)
    criterion = build_criterion()

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss = validate_epoch(model, val_loader, criterion)
        checkpoint_path = save_checkpoint(model, epoch, run_dir)
        print(f"Epoch {epoch:02d}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Checkpoint saved at: {checkpoint_path}")

if __name__ == "__main__":
    from tokenizer.character_tokenizer import CharacterTokenizer
    tokenizer = CharacterTokenizer()
    tokenizer.fit(open("data/fantasy_names.csv").read().splitlines()[1:])
    train("data/fantasy_names.csv", tokenizer, "checkpoints/run2")