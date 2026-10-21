import torch
from model.lstm.lstm import NameLSTM

def load_model(checkpoint_path, vocab_size, embed_dim=32, hidden_dim=64):
    model = NameLSTM(vocab_size, embed_dim, hidden_dim)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    return model

def encode_prompt(prompt, tokenizer):
    bos_id = tokenizer._token_to_id["<BOS>"]
    token_ids = [bos_id] + tokenizer.encode(prompt)
    return torch.tensor(token_ids).unsqueeze(0)

def generate(model, tokenizer, prompt, temperature=1.0, max_length=50):
    eos_id = tokenizer._token_to_id["<EOS>"]
    input_ids = encode_prompt(prompt, tokenizer)

    generated= []

    for _ in range(max_length):
        with torch.no_grad():
            output = model(input_ids)
        
        logits = output[0, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()

        pad_id = tokenizer._token_to_id["<PAD>"]
        bos_id = tokenizer._token_to_id["<BOS>"]

        if next_id in (eos_id, pad_id, bos_id):
            break

        generated.append(next_id)
        input_ids = torch.cat([input_ids, torch.tensor([[next_id]])], dim=1)

    return tokenizer.decode(generated)

if __name__ == "__main__":
    from tokenizer.character_tokenizer import CharacterTokenizer

    tokenizer = CharacterTokenizer()
    tokenizer.fit(open("data/fantasy_names.csv").read().splitlines()[1:])

    model = load_model("checkpoints/epoch_50.pt", vocab_size=tokenizer.vocabulary_size)
    print(generate(model, tokenizer, prompt="elf:"))
