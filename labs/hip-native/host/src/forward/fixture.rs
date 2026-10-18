//! A small numerical smoke input, not a replacement for the pinned real-model acceptance.
use std::fs::OpenOptions;
use std::path::Path;

use gguf::write::{Layout, TensorDecl, Writer};
use gguf::{GgmlType, Value};
use model::arch::{names, qwen35moe::names as qnames};

const HIDDEN: u64 = 256;
const VOCAB: u64 = 258;

struct Random(u32);
impl Random {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        self.0
    }
}

fn packed(kind: GgmlType, k: u64, rows: u64, rng: &mut Random) -> Vec<u8> {
    let block = kind.blck_size().expect("fixture type has a block size");
    let bytes = kind.type_size().expect("fixture type has a byte size") as usize;
    let mut out = vec![0; (k / block * rows) as usize * bytes];
    for chunk in out.chunks_exact_mut(bytes) {
        match kind {
            GgmlType::F32 => {
                let value = ((rng.next() % 1024) as f32 - 512.0) / 8192.0;
                chunk.copy_from_slice(&value.to_le_bytes());
            }
            GgmlType::Q8_0 => {
                chunk[..2].copy_from_slice(&0x2000_u16.to_le_bytes());
                for code in &mut chunk[2..] {
                    *code = ((rng.next() % 255) as i16 - 127) as i8 as u8;
                }
            }
            GgmlType::Q4_K => {
                chunk[..2].copy_from_slice(&0x2400_u16.to_le_bytes());
                chunk[2..4].copy_from_slice(&0x2400_u16.to_le_bytes());
                chunk[4..8].fill(1);
                chunk[8..12].fill(8);
                chunk[12..16].fill(0x81);
                for code in &mut chunk[16..] {
                    *code = rng.next() as u8;
                }
            }
            GgmlType::Q6_K => {
                for code in &mut chunk[..192] {
                    *code = rng.next() as u8;
                }
                chunk[192..208].fill(1);
                chunk[208..].copy_from_slice(&0x1c00_u16.to_le_bytes());
            }
            _ => unreachable!("fixture only generates the admitted formats"),
        }
    }
    out
}

pub(super) struct Fixture {
    pub kvs: Vec<(String, Value)>,
    pub tensors: Vec<(TensorDecl, Vec<u8>)>,
}

impl Fixture {
    pub fn new() -> Self {
        let mut kvs = vec![
            (
                "general.architecture".into(),
                Value::String("qwen35".into()),
            ),
            (
                "general.name".into(),
                Value::String("HIP bounded hybrid numerical smoke".into()),
            ),
            ("tokenizer.ggml.model".into(), Value::String("gpt2".into())),
            ("tokenizer.ggml.pre".into(), Value::String("qwen2".into())),
            ("tokenizer.ggml.bos_token_id".into(), Value::U32(256)),
            ("tokenizer.ggml.eos_token_id".into(), Value::U32(256)),
            ("tokenizer.ggml.add_bos_token".into(), Value::Bool(false)),
            (
                "tokenizer.ggml.merges".into(),
                Value::Array(vec![Value::String("a b".into())]),
            ),
        ];
        let mut byte_alphabet: Vec<u32> = (33..=126).chain(161..=172).chain(174..=255).collect();
        let mut extra = 0;
        let tokens: Vec<_> = (0..256)
            .map(|byte| {
                let cp = if byte_alphabet.contains(&byte) {
                    byte
                } else {
                    byte_alphabet.push(byte);
                    let cp = 256 + extra;
                    extra += 1;
                    cp
                };
                Value::String(char::from_u32(cp).expect("GPT2 byte alphabet").to_string())
            })
            .chain([
                Value::String("<|endoftext|>".into()),
                Value::String("ab".into()),
            ])
            .collect();
        kvs.push(("tokenizer.ggml.tokens".into(), Value::Array(tokens)));
        kvs.push((
            "tokenizer.ggml.token_type".into(),
            Value::Array(
                (0..VOCAB)
                    .map(|i| Value::I32(if i == 256 { 3 } else { 1 }))
                    .collect(),
            ),
        ));
        for (key, value) in [
            ("block_count", 2),
            ("embedding_length", HIDDEN),
            ("vocab_size", VOCAB),
            ("context_length", 64),
            ("attention.head_count", 2),
            ("attention.head_count_kv", 1),
            ("attention.key_length", 128),
            ("attention.value_length", 128),
            ("rope.dimension_count", 64),
            ("feed_forward_length", 256),
            ("ssm.conv_kernel", 4),
            ("ssm.state_size", 128),
            ("ssm.time_step_rank", 2),
            ("ssm.group_count", 1),
            ("ssm.inner_size", 256),
            ("full_attention_interval", 2),
        ] {
            kvs.push((format!("qwen35.{key}"), Value::U32(value as u32)));
        }
        kvs.push(("qwen35.rope.freq_base".into(), Value::F32(10000.0)));
        kvs.push((
            "qwen35.attention.layer_norm_rms_epsilon".into(),
            Value::F32(1.0e-6),
        ));
        kvs.push((
            "qwen35.rope.dimension_sections".into(),
            Value::Array([8, 12, 12, 0].map(Value::U32).to_vec()),
        ));
        let mut this = Self {
            kvs,
            tensors: Vec::new(),
        };
        let mut rng = Random(0x87ab_35f1);
        this.matrix(names::token_embd(), HIDDEN, VOCAB, GgmlType::Q8_0, &mut rng);
        // A signed head gain prevents tied random embeddings from making this an identity predictor.
        let head_gain: Vec<_> = (0..HIDDEN)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        this.floats(names::output_norm(), vec![HIDDEN], &head_gain);
        for l in 0..2 {
            this.gain(names::blk(l, "attn_norm.weight"), HIDDEN);
            this.gain(names::blk(l, "post_attention_norm.weight"), HIDDEN);
            if l == 0 {
                this.matrix(qnames::attn_qkv(l), HIDDEN, 512, GgmlType::Q6_K, &mut rng);
                this.matrix(qnames::attn_gate(l), HIDDEN, 256, GgmlType::Q4_K, &mut rng);
                this.matrix(qnames::ssm_alpha(l), HIDDEN, 2, GgmlType::Q4_K, &mut rng);
                this.matrix(qnames::ssm_beta(l), HIDDEN, 2, GgmlType::F32, &mut rng);
                this.matrix(qnames::ssm_out(l), 256, HIDDEN, GgmlType::Q4_K, &mut rng);
                let conv: Vec<f32> = (0..512)
                    .flat_map(|c| {
                        [-0.12, 0.18, 0.3, 0.6].map(|v| v * (0.9 + (c % 9) as f32 * 0.025))
                    })
                    .collect();
                this.floats(qnames::ssm_conv1d(l), vec![4, 512], &conv);
                this.floats(qnames::ssm_dt_bias(l), vec![2], &[-0.3, 0.2]);
                this.floats(qnames::ssm_a(l), vec![2], &[-0.7, -1.1]);
                this.gain(qnames::ssm_norm(l), 128);
            } else {
                this.matrix(qnames::attn_q(l), HIDDEN, 512, GgmlType::Q4_K, &mut rng);
                this.matrix(qnames::attn_k(l), HIDDEN, 128, GgmlType::Q4_K, &mut rng);
                this.matrix(qnames::attn_v(l), HIDDEN, 128, GgmlType::Q6_K, &mut rng);
                this.matrix(
                    qnames::attn_output(l),
                    256,
                    HIDDEN,
                    GgmlType::Q4_K,
                    &mut rng,
                );
                this.gain(qnames::attn_q_norm(l), 128);
                this.gain(qnames::attn_k_norm(l), 128);
            }
            this.matrix(
                names::blk(l, "ffn_gate.weight"),
                HIDDEN,
                256,
                GgmlType::Q4_K,
                &mut rng,
            );
            this.matrix(
                names::blk(l, "ffn_up.weight"),
                HIDDEN,
                256,
                GgmlType::Q4_K,
                &mut rng,
            );
            this.matrix(
                names::blk(l, "ffn_down.weight"),
                256,
                HIDDEN,
                GgmlType::Q6_K,
                &mut rng,
            );
        }
        this
    }

    fn matrix(&mut self, name: String, k: u64, rows: u64, kind: GgmlType, rng: &mut Random) {
        let bytes = packed(kind, k, rows, rng);
        self.tensors.push((
            TensorDecl {
                name,
                dims: vec![k, rows],
                type_id: kind.as_u32(),
                nbytes: bytes.len() as u64,
            },
            bytes,
        ));
    }

    fn floats(&mut self, name: String, dims: Vec<u64>, values: &[f32]) {
        let bytes: Vec<_> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.tensors.push((
            TensorDecl {
                name,
                dims,
                type_id: 0,
                nbytes: bytes.len() as u64,
            },
            bytes,
        ));
    }

    fn gain(&mut self, name: String, n: u64) {
        let values: Vec<_> = (0..n)
            .map(|i| 1.0 + ((i % 7) as f32 - 3.0) * 0.025)
            .collect();
        self.floats(name, vec![n], &values);
    }

    pub fn write(self, path: &Path) -> Result<(), String> {
        let declarations = self.tensors.iter().map(|(t, _)| t.clone()).collect();
        let layout = Layout::new(&self.kvs, declarations).map_err(|e| e.to_string())?;
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .map_err(|e| e.to_string())?;
        let mut writer = Writer::new(file, layout).map_err(|e| e.to_string())?;
        for (tensor, bytes) in self.tensors {
            writer
                .tensor(&tensor.name, &bytes)
                .map_err(|e| e.to_string())?;
        }
        writer.finish().map_err(|e| e.to_string())?;
        Ok(())
    }
}
