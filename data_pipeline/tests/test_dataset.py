"""
Tests for NameDataset

It covers loading, parsing, encoding and error handling
All the tests use an in-memory CSV via io.StringIO to avoid disk dependencies
"""

import io
import warnings
import pytest
from tokenizer.character_tokenizer import CharacterTokenizer
from data_pipeline import NameDataset

SAMPLE_CSV = """name,race
elf:Aeloria,elf
human:Arthas,human
dwarf:Bromir,dwarf
halfling:Rosie,halfling
elf:Ae:loria,elf
elf:  Aeloria ,elf
invalidrow,
"""

@pytest.fixture
def tokenizer():
    t = CharacterTokenizer()
    lines = SAMPLE_CSV.strip().splitlines()
    names = [line.split(",")[0] for line in lines[1:] if line.strip()]
    t.fit(names)
    return t

@pytest.fixture
def dataset(tokenizer):
    return NameDataset(io.StringIO(SAMPLE_CSV), tokenizer)

def test_header_is_skipped(dataset, tokenizer):
    header_encoded = [CharacterTokenizer.BOS]+tokenizer.encode("name")+ [CharacterTokenizer.EOS] 
    first_sequence = [t for t in dataset[0] if t !=CharacterTokenizer.PAD]   
    assert first_sequence!=header_encoded

def test_valid_rows_load_correctly(dataset, tokenizer):
    sequence = dataset[0]
    assert sequence[0] == CharacterTokenizer.BOS

def test_original_casing_preserved(dataset, tokenizer):
    """Casing is preserved — no lowercasing or uppercasing is applied."""
    original = "elf:Aeloria"
    encoded_original = tokenizer.encode(original)
    body = [t for t in dataset[0] if t != CharacterTokenizer.BOS and t != CharacterTokenizer.EOS and t!=CharacterTokenizer.PAD]
    assert body == encoded_original

def test_len(dataset):
    """__len__ returns the number of valid sequences."""
    assert len(dataset) == 6

def test_getitem_returns_list_of_ints(dataset):
    sequence = dataset[0]
    assert isinstance(sequence, list)
    assert all(isinstance(i, int) for i in sequence)

def test_getitem_out_of_range(dataset):
    with pytest.raises(IndexError):
        _ = dataset[len(dataset)]

def test_malformed_rows_skipped_with_warning(tokenizer):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        NameDataset(io.StringIO(SAMPLE_CSV), tokenizer)
    assert any("invalidrow" in str(w.message).lower() or "malformed" in str(w.message).lower() for w in caught), "Expected a warning about malformed rows"

def test_empty_file_raises_error(tokenizer):
    empty_csv = "name, race\n"
    with pytest.raises(ValueError):
        NameDataset(io.StringIO(empty_csv), tokenizer)

def test_bos_is_first_token(dataset):
    for i in range(len(dataset)):
        assert dataset[i][0] == CharacterTokenizer.BOS, \
            f"Sequence {i} does not start with BOS"

def test_eos_is_last_non_pad_token(dataset):
    for i in range(len(dataset)):
        sequence = dataset[i]
        non_pad = [t for t in sequence if t != CharacterTokenizer.PAD]
        assert non_pad[-1] == CharacterTokenizer.EOS, \
            f"Sequence {i} does not end with EOS before padding"
        
def test_white_space_is_stripped(dataset, tokenizer):
    """Leading and trailing whitespace should be stripped from the name before encoding."""
    original = "elf:  Aeloria "
    encoded_original = tokenizer.encode("elf:Aeloria")
    expected_body = [CharacterTokenizer.BOS] + encoded_original + [CharacterTokenizer.EOS]
    matches = []
    for i in range(len(dataset)):
        sequence = dataset[i]
        body = [t for t in sequence if t != CharacterTokenizer.PAD]
        if body == expected_body:
            matches.append(body)
    assert len(matches)==2, "Whitespace was not stripped correctly"

def test_split_on_first_colon(dataset, tokenizer):
    colon_id = tokenizer.encode(":")[0]
    for i in range(len(dataset)):
        sequence = dataset[i]
        body = [t for t in sequence if t != CharacterTokenizer.PAD and t != CharacterTokenizer.BOS and t != CharacterTokenizer.EOS]
        first_colon_pos = body.index(colon_id)
        name_tokens = body[first_colon_pos+1:]
        if colon_id in name_tokens:
            return
    pytest.fail("No sequence found where name part contains a colon")

def test_all_sequences_same_length(dataset):
    """All sequences should be padded to the same length."""
    lengths = [len(dataset[i]) for i in range(len(dataset))]
    assert len(set(lengths)) == 1, "Not all sequences are the same length"

def test_padding_is_on_the_right(dataset):
    for i in range(len(dataset)):
        sequence = dataset[i]
        if CharacterTokenizer.PAD in sequence:
            first_pad = sequence.index(CharacterTokenizer.PAD)
            assert all(t == CharacterTokenizer.PAD for t in sequence[first_pad + 1:]), \
                f"Sequence {i} is not padded on the right"
            
def test_empty_race_or_name_skipped_with_warning(tokenizer):
    csv_with_empty = "name,race\n:Aeloria,elf\nelf:,elf\n"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError):
            NameDataset(io.StringIO(csv_with_empty), tokenizer)
    assert any("malformed" in str(w.message).lower() for w in caught)

def test_file_path_support(tokenizer, tmp_path):
    file_path = tmp_path / "test.csv"
    file_path.write_text(SAMPLE_CSV)
    dataset_from_file = NameDataset(str(file_path), tokenizer)
    dataset_from_stringio = NameDataset(io.StringIO(SAMPLE_CSV), tokenizer)
    assert len(dataset_from_file) == 6