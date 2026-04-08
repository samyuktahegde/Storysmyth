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
└── 📁storysmyth
    └── README.md
```

## Components
### Tokenizer
Converts raw name strings into sequences of tokens for the model, and converts the model's token output back into readable names. The current implementation is character-level tokenization —> each unique character in the corpus gets its own token ID.
### Data Pipeline
Planned
### Model
Planned
### Training
Planned
