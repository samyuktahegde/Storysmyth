import json

class CharacterTokenizer:

    def __init__(self):
        self._token_to_id = {}
        self._id_to_token = {}
        self._fitted = False

    def fit(self, corpus):
        unique_characters = sorted(set("".join(corpus)))
        self._token_to_id = {char: idx for idx, char in enumerate(unique_characters)}
        self._id_to_token = {idx: char for char, idx in self._token_to_id.items()}
        self._fitted = True

    def get_vocabulary(self):
        return dict(self._token_to_id)

    def _require_fitted(self):
        if not self._fitted:
            raise RuntimeError("Tokenizer is not fitted yet.")
        
    def encode(self, text):
        self._require_fitted()
        return [self._token_to_id[char] for char in text]
    
    def decode(self, ids):
        self._require_fitted()
        return "".join([self._id_to_token[id] for id in ids])
    
    def save(self, path:str) -> "CharacterTokenizer":
        self._require_fitted()
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"token_to_id": self._token_to_id}, f, ensure_ascii=False, indent=2)
        return self
    
    @classmethod
    def load(cls, path: str) -> "CharacterTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        tokenizer = cls()
        tokenizer._token_to_id = payload["token_to_id"]
        tokenizer._id_to_token = {int(idx): char for char, idx in payload["token_to_id"].items()}
        tokenizer._fitted = True
        return tokenizer

    @property
    def vocabulary_size(self) -> int:
        self._require_fitted()
        return len(self._token_to_id)
    