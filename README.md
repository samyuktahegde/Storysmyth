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
1     |  Tokenizer     |  Convert names to integers and back | Planned
2     |  Data Pipeline      |  Load, clean, encode, and batch training data |  Planned    


## Project Structure
```
└── 📁Storysmyth
    └── 📁data
        ├── fantasy_names.csv
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
Planned
### Model
Planned
### Training
Planned
