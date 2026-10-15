# Fixed dense27B inputs

These literal input IDs were produced from the exact Qwen3.8-27B model named in
identity.json. corpus.txt is synthetic prose made for this verification. p64.ids
and p4096.ids are the complete prompts; each corresponding f*.ids holds the next
32 teacher-forced IDs. No tokenization/chat template is applied during captures.
Use p64.ids/p4096.ids as --short/--long for dense-serve.py; an early EOS remains
an incomplete fixed-count diagnostic. See the source PR's handoff documentation.
