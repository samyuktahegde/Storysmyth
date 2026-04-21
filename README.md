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
1     |  Tokenizer     |  Convert names to integers and back | Done
2     |  Data Pipeline      |  Load, clean, encode, and batch training data |  Planned    


## Project Structure
```
└── 📁Storysmyth
    └── 📁data
        ├── fantasy_names.csv
    └── 📁data-pipeline
        ├── __ini__.py
        └── dataset.py
    └── 📁tokenizer
        └── 📁tests
            ├── __init__.py
            ├── test_character_tokenizer.py
        ├── __init__.py
        └── character_tokenizer.py
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
- Vocabulary is saved as a json file `token_to_id`
### Data Pipeline
Loads the names from `data/fantasy_names.csv`, encodes them using the tokenizer and prepares them as fixed length token ID sequences for training.
#### Design Decisions:
- Special Toekns PAD, BOS and EOS are intrduced at this stage. They are defined as constants in CharacterToeknizer class with reserved IDs - `PAD=0`, `BOS=1` and `EOS=2`
- `fit()` reserves the IDs 0, 1 and 2 for special tokens before building the vocabulary and then regular tokens are assigned IDs from 3.
- `race: name` format is kept as is. The model learns that after `:` character names appear.
- Each sequence is formatted as `encode([BOS]+race:name+[EOS])`
- All sequences are padded with `PAD` on the right to match the longest sequence in the dataset.
- The output is a list of equal length Python lists of integers.
### Model
Planned
### Training
Planned
