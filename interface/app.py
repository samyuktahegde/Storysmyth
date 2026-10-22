import argparse
import os
from tokenizer.character_tokenizer import CharacterTokenizer
from generator.generator import load_model, generate

DEFAULT_RUN = "checkpoints/run1"
DEFAULT_EPOCH = 50


def build_parser():
    parser = argparse.ArgumentParser(description="Generate a fantasy name.")
    parser.add_argument("--race", required=True, help="Race prefix, e.g. elf, orc, dwarf")
    parser.add_argument("--run", default=DEFAULT_RUN)
    parser.add_argument("--epoch", type=int, default=DEFAULT_EPOCH)
    parser.add_argument("--temperature", type=float, default=1.0)
    return parser

def get_checkpoint_path(run, epoch):
    return os.path.join(run, f"epoch_{epoch:02d}.pt")

def load_tokenizer(run):
        return CharacterTokenizer.load(os.path.join(run, "tokenizer.json"))

def main(argv=None):
    args = build_parser().parse_args(argv)
    checkpoint_path = get_checkpoint_path(args.run, args.epoch)
    tokenizer = load_tokenizer(args.run)
    model = load_model(checkpoint_path, tokenizer.vocabulary_size)
    name = generate(model, tokenizer, f"{args.race}:", temperature=args.temperature)
    print(name)

if __name__ == "__main__":
    main()