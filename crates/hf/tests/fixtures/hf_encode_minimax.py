"""The Hugging Face encode corpus for MiniMax-M3.1-Flash's folder tokenizer (crates/hf/tests/minimax_encode.rs).

usage (Python 3 with tokenizers==0.21.0 from PyPI):
    python3 hf_encode_minimax.py <folder> <prompts dir> | gzip -9n > minimax-hf-encode.json.gz

<folder> holds tokenizer.json of MiniMaxAI/MiniMax-M3.1-Flash at REV (MiniMax-M3's is the same file). <prompts dir> holds
a.txt .. e.txt, the texts behind tools/ref/data/minimaxm3/{a..e}.ids (llama-tokenize's ids decode to their bytes). Each case
is a text and the ids `Tokenizer.encode(text, add_special_tokens=False)` gives. The texts are those of hf_encode.py's
corpus (every part of the pipeline: contractions in every case and fold, letters with marks, spaces, numbers, Korean, code,
emoji, codepoints of every Unicode version, NFC's reordering), the five oracle prompts, and every added token of the
folder alone, between letters, back to back with every other, and spelled over another where two share their edge (the
texts where a longest-first split and tokenizers' leftmost one differ), and the regex's letter categories against each other:
every pair of a hundred codepoints (Lu, Lt, Ll, Lm, Lo, M, N, punctuation, space) alone and between letters, and words that
change case in scripts that have it.
"""
import json
import sys
from pathlib import Path

import hf_corpus
import hf_encode

REV = "76faee1b4ac0f412e547781172e94548a6526a88"


def added_token_texts(added):
    contents = [a["content"] for a in added]
    out = []
    for c in contents:
        out += [c, f"a{c}b", f" {c} ", f"{c}\n{c}"]
    for a in contents:
        for b in contents:
            out.append(a + b)
            for at in range(1, len(a)):
                if b.startswith(a[at:]) and len(a) - at < len(b):
                    out += [a[:at] + b, "x" + a[:at] + b + "y", a[:at] + b + b[len(a) - at:] + a]
    return out


# Lu, Lt, Ll, Lm, Lo, M, N and the rest, ASCII and not, scripts with case (Latin, Greek, Cyrillic, Armenian, Georgian)
# and without (CJK, Hangul, Thai, Devanagari).
REPRESENTATIVES = list(
    "aZmQ\u00e9\u00c9\u00df\u00ff\u01c5\u01c6\u01c4\u03b1\u0391\u03c2\u044f\u042f\u0561\u0531\u10d0\u1c90"
    "\u02b0\u02c8\u30fc\u4e2d\u3042\ud55c\u0e01\u0928\u0301\u094d\u0e31\u20dd\u0903\u1d2c"
    "1\u0663\u00b2\u2160.'-_ \n\t/\u00a0\u3000"
)


def case_texts():
    out = []
    for a in REPRESENTATIVES:
        for b in REPRESENTATIVES:
            out += [a + b, f"x{a}{b}y", f" {a}{b}", f"{a}{b}{a}"]
    out += ["\u041f\u0440\u0438\u0432\u0435\u0442\u041c\u0438\u0440", "caf\u00e9\u00c9cole", "\u00c9cole\u00c9cole",
            "\u0391\u03bb\u03c6\u03b1\u0392\u03ae\u03c4\u03b1", "\u01c5ungle", "\u02b0ello", "HELLO\u00c9", "\u00e9\u00c9\u00e9\u00c9",
            " \u041f\u0440\u0438\u0432\u0435\u0442 \u041c\u0438\u0440", "\u00e9's", "\u00c9'S", "\u4e2d's",
            "\u041f\u0440\u0438\u0432\u0435\u0442'\u017f", "\u00c9COLE'ST", "\u0561\u0562\u0531\u0532\u0561"]
    return out


def corpus(prompts, added):
    texts = [(prompts / f"{n}.txt").read_bytes().decode("utf-8") for n in "abcde"]
    return list(dict.fromkeys(texts + hf_encode.corpus() + added_token_texts(added) + case_texts()))


def main():
    folder, prompts = Path(sys.argv[1]), Path(sys.argv[2])
    data, tok = hf_corpus.read_tokenizer(folder)
    texts = corpus(prompts, json.loads(data)["added_tokens"])
    hf_corpus.emit(data, f"MiniMaxAI/MiniMax-M3.1-Flash@{REV}/tokenizer.json", hf_corpus.encoded(tok, texts))


if __name__ == "__main__":
    main()
