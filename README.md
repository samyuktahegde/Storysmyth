# Storysmyth


## Project Overview
Storysmyth is a fantasy name generator driven by a small LLM.
It takes a corpus of fantasy names, learns the patterns and generates similar names belonging to that fantasy world.

## Project Goals

- Build a tiny language model from scratch
- Train it on a corpus of fantasy names
- Generate new names that are relevant to the fantasy world
- Keep the codebase simple, readable and easy to extend

## Roadmap

Phase | Component     |  Description    | Status
|---|---|---|---|
1     |  Tokenizer     |  Convert names to integers and back | ✅ Done
2     |  Data Pipeline      |  Load, clean, encode, and batch training data | ✅ Done  
3     |  LSTM Model      | Character-level LSTM Language Model |   ✅ Done 
4     |  Transformer Model      | - | -
5     |  Training      | Train the model on the name corpus | 🔄 In Progress
6     |  Generator      | - | -
7     |  Interface      | - | -


## Project Structure
```
└── 📁Storysmyth
    └── 📁data
        ├── fantasy_names.csv
    └── 📁data_pipeline
        ├── __ini__.py
        └── dataset.py
    └── 📁tokenizer
        └── 📁tests
            ├── __init__.py
            ├── test_character_tokenizer.py
        ├── __init__.py
        └── character_tokenizer.py
    └── 📁data_pipeline
        └── 📁tests
            ├── __init__.py
            ├── test_dataset.py
        ├── __init__.py
        └── dataset.py
    └── 📁model
        └── 📁lstm
            └── 📁tests
                ├── __init__.py
                ├── test_lstm.py
            ├── __init__.py
            ├── lstm.py
        └── __init__.py
    └── .gitignore
    └── README.md
```

## Data
Training data is located in `data/fantasy_names.csv`. It contains fantasy names curated with the help of an AI tool(ChatGPT).
It includes synthetically generated names of humans, elves, dwarves and halflings.
The file has two columns,
Columns | Description | Example
|---|---|---|
`name` | Fantasy name with race as its prefix | `elf:Loia`
`race`  | Race of the character | `elf`

The `name` column includes a race prefix separated by a colon from the character name so that it becomes easy to condition the model while training. The `elf:` prefix will produce elvish sounding names.

## Usage
### Run tests
```
bash
python3 -m pytest tokenizer/tests -v
python3 -m pytest data_pipeline/tests -v
```

## Components
### Tokenizer
Converts raw name strings into sequences of tokens for the model, and converts the model's token output back into readable names. The current implementation is character-level tokenization —> each unique character in the corpus gets its own token ID.
#### Design Decisions:
- Character-level - There is one token per character
- The vocabulary is built by scanning the corpus once and sorting the characters alphabetically
- Case Sensitivity - A and a are distinct characters
- Special characters are treated the same as letters
- There are no special tokens at this stage
- Encoding raises a KeyError on unknown characters
- Vocabulary is saved as a json file `token_to_id`. `id_to_token` is reconstructed on load.
### Data Pipeline
Loads the names from `data/fantasy_names.csv`, encodes them using the tokenizer and prepares them as fixed length token ID sequences for training.
#### Design Decisions:
- CSV header row is skipped by default.
- Special Tokens PAD, BOS and EOS are intrduced at this stage. They are defined as constants in CharacterTokenizer class with reserved IDs - `PAD=0`, `BOS=1` and `EOS=2`
- `fit()` reserves the IDs 0, 1 and 2 for special tokens before building the vocabulary and then regular tokens are assigned IDs from 3.
- `race: name` format is kept as is. The model learns that after `:` character names appear. They are split on the first colon only thus retaining any colons within a name.
- Each sequence is formatted as `encode([BOS]+race:name+[EOS])`
- All sequences are padded with `PAD` on the right to match the longest sequence in the dataset.
- Malformed rows are skipped with a warning.
- An empty file or a file with no valid rows raises an error at construction time.
- As of now, batching is out of scope. The `NameDataSet` prepares individual sequences.
- There is no PyTorch dependency at this stage. `__len__` and `__getitem__` mirror the interface DataLoader will eventually expect.
- The output is a list of equal length Python lists of integers.
### LSTM Model
A character-level LSTM language model hat takes a sequence of token IDs and predicts the next token at each position. This is a baseline model before the transformer is introduced in Phase 4.
#### Design Decisions:
- PyTorch is introduced as a dependency starting in this phase. Phases 1 and 2 remain in pure Python.
- NameLSTM subclasses torch.nn.module.
- Three components are wired in sequence - embedding layer-> LSTM layer -> linear projection.
- Configurable hyperparameters at construction time: vocab_size, embed_dim, hidden_dim, num_layers.
- num_layers defaults to 1.
- Output is raw logits of shape (batch_size, seq_len, vocab_size) - softmax and loss are left to the training loop.
- Hidden state is initialised to zeros at the start of each forward pass.

### Transformer Model
### Training
Trains NameLSTM on the name corpus, evaluates on a validation split after each epoch, and saves model weights to disk.
#### Design Decisions:
- Loss function is nn.CrossEntropyLoss — standard for next token prediction
- Optimiser is Adam — reliable default that adapts learning rate per parameter, works well for character-level models without extensive tuning
- Dataset is split 90/10 into train and validation sets
- Batch size, number of epochs, and learning rate are configurable — defaults are 32, 50, and 0.001 respectively
- DataLoader is introduced here to handle batching and shuffling
- Model weights are saved to checkpoints/ at the end of each epoch
- TDD where it makes sense — loss computation and checkpoint saving are tested; the loop as a whole is not

### Generator

### Interface

