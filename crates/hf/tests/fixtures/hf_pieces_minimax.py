"""The pre-tokenizer words of Hugging Face's `tokenizers` for MiniMax-M3.1-Flash's tokenizer.json
(crates/tokenizer, `minimax_m2_hf_pieces_are_hugging_faces`).

usage (Python 3 with tokenizers==0.21.0 from PyPI):
    python3 hf_pieces_minimax.py <folder> <prompts dir> | gzip -9n > minimax-hf-pieces.json.gz

The texts are hf_encode_minimax.py's, but those that spell an added token (which `tokenizers` cuts out before its
pre-tokenizer runs). Each case is a text and the byte length of each piece `pre_tokenizer.pre_tokenize_str` gives, which
is the number of ByteLevel characters of the piece. Ids do not show a split that no merge crosses; the pieces do.
"""
import json
import sys
from pathlib import Path

import hf_corpus
import hf_encode_minimax as encode


def main():
    folder, prompts = Path(sys.argv[1]), Path(sys.argv[2])
    data, tok = hf_corpus.read_tokenizer(folder)
    added = json.loads(data)["added_tokens"]
    spellings = [a["content"] for a in added]
    texts = [t for t in encode.corpus(prompts, added) if not any(s in t for s in spellings)]
    cases = ({"text": t, "pieces": [len(p) for p, _ in tok.pre_tokenizer.pre_tokenize_str(t)]} for t in texts)
    hf_corpus.emit(data, f"MiniMaxAI/MiniMax-M3.1-Flash@{encode.REV}/tokenizer.json", cases)


if __name__ == "__main__":
    main()
