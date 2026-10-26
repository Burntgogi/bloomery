# bloomery @VERSION@ — prebuilt Apple Silicon release

bloomery's Metal backend: a Rust host and Metal kernels that serve a mixture-of-experts model on a Mac's GPU, with
llama-server's HTTP API. Built from commit @COMMIT@. Source, measurements and the full README:
https://github.com/midagedev/bloomery (the Mac page: `docs/mac.md`).

## What is in this archive

| File | What it does |
|---|---|
| `bin/bloomery-serve-metal` | llama-server's HTTP API (`/v1/chat/completions`, `/completion`, `/v1/models`, `/health`) on the Mac's GPU |
| `bin/generate_metal` | one prompt from the command line; prints the rates on stderr |
| `bin/prepare` | converts an MLX 4-bit checkpoint folder (mlx-community) into a GGUF split once |

## Requirements

- An Apple Silicon Mac with macOS 26.2 or newer; an older macOS is refused by name at the start.
- One model architecture runs so far: Qwen3.6-35B-A3B. A 48 GB Mac is tested; a 32 GB Mac is untested.
- The kernels compile from their embedded source on the first start of each process; no Xcode is needed.

## Run

```sh
# a Q4_K_M GGUF opens directly (for example lmstudio-community/Qwen3.6-35B-A3B-GGUF)
bin/bloomery-serve-metal -m PATH/Qwen3.6-35B-A3B-Q4_K_M.gguf --port 8080

# or convert mlx-community/Qwen3.6-35B-A3B-4bit once, then serve the split
bin/prepare PATH/Qwen3.6-35B-A3B-4bit OUT/qwen36-mlx4
bin/bloomery-serve-metal -m OUT/qwen36-mlx4-00001-of-00001.gguf --port 8080

curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "What is the capital of France?"}], "max_tokens": 256}'
```

The model thinks first: its reasoning comes back in `reasoning_content`, so leave `max_tokens` room for the answer.
The server takes `-m`, `--host`, `--port`, `--ctx` (`--ctx-size`, `-c`), `--chat-template-file`, `--alias`,
`--queue-depth`, `--api-key` and `--api-key-file`; any other flag is refused by name with exit 64. It serves one
sequence, text only.

If macOS says the binary "cannot be opened" after a browser download, clear the quarantine flag:
`xattr -d com.apple.quarantine bin/*`.

MIT licensed (`LICENSE`); third-party notices in `THIRD_PARTY_NOTICES.md`.
