"""What the Hugging Face corpus writers share (hf_encode.py, hf_encode_minimax.py, hf_pieces_minimax.py): a folder's
tokenizer.json as `tokenizers` reads it, the header every corpus opens with, and the one shape each writes to stdout.
A writer is run as a script from anywhere; it finds this file and its siblings by `import`, as Python puts the script's
own directory first on the path.
"""
import hashlib
import json
import sys
from pathlib import Path

import tokenizers
from tokenizers import Tokenizer


def read_tokenizer(folder):
    """The bytes of `<folder>/tokenizer.json` and the Tokenizer they make."""
    data = (Path(folder) / "tokenizer.json").read_bytes()
    return data, Tokenizer.from_str(data.decode("utf-8"))


def encoded(tok, texts):
    """The cases of an encode corpus: each text and the ids `tok` gives it."""
    return ({"text": t, "ids": tok.encode(t, add_special_tokens=False).ids} for t in texts)


def emit(data, source, cases, **head):
    """Write the corpus to stdout: the header (the `tokenizers` version, `head`, `source` and the sha256 of the
    tokenizer.json bytes `data`) and `cases`, one JSON object a line."""
    head = {"tokenizers": tokenizers.__version__, **head, "source": source, "sha256": hashlib.sha256(data).hexdigest()}
    lines = [json.dumps(c, ensure_ascii=False) for c in cases]
    sys.stdout.write(json.dumps(head, ensure_ascii=False)[:-1] + ', "cases": [\n' + ",\n".join(lines) + "\n]}\n")
