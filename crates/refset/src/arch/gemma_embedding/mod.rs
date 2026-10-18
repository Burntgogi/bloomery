//! EmbeddingGemma v1 text rows, pooled vectors and llama-server vectors.
use crate::RefError;
use crate::emb2::{Kind, LCPP_BUILD, Profile, family};
use crate::family::{Build, Family, Identity};

/// The Q8_0 file from ggml-org/embeddinggemma-300M-GGUF, revision 0f741b5a.
pub const MODEL: &str = "/models/embeddinggemma-300M/embeddinggemma-300M-Q8_0.gguf";
/// The file's SHA256, checked against the Hub LFS digest.
pub const MODEL_SHA256: &str = "b5ce9d77a3fc4b3b39ccb5643c36777911cc4eb46a66962eadfa3f5f60490d63";
/// The architecture in the model and its sets.
pub const ARCH: &str = "gemma-embedding";
/// The text-only model has no projector.
pub static PROFILE: Profile = Profile {
    model: MODEL,
    model_sha256: MODEL_SHA256,
    mmproj: None,
};
fn model() -> Result<String, RefError> {
    Ok(MODEL.to_string())
}
/// The SentencePiece ids, including BOS and EOS in both parse modes, from the oracle commit's `llama-tokenize`. The set
/// names that commit in its `# build` line, so any machine's build of it is the same oracle: the executable's and the library's
/// md5s change with every rebuild and are not the identity.
// PIN(2026-10-18): the oracle commit, not one machine's llama-tokenize and libllama.so md5s, which no other build of it matched.
pub static TOKENIZER: Family = Family {
    name: "tokenizer-llama",
    sets: &["ref-tokenizer-llama"],
    resolve: None,
    recipe: "just dump-ref-tokenizer llama",
    identity: Identity::Vocabulary,
    arch: None,
    build: Some(Build::Is(LCPP_BUILD)),
    runs: Some(model),
    draft_runs: None,
    consumers: &["gate-tokenizer"],
};

const RECIPE: &str = "just dump-ref-emb1";
/// Token rows, pooled vectors and layer taps on the card.
pub static TEXT: Family = family(
    "text-gemma-embedding",
    &["ref_gemma_embedding_text"],
    Kind::Text,
    &PROFILE,
    ARCH,
    model,
    RECIPE,
);
/// The text set with no card in view.
pub static TEXT_CPU: Family = family(
    "text-gemma-embedding-cpu",
    &["ref_gemma_embedding_text.cpu"],
    Kind::TextCpu,
    &PROFILE,
    ARCH,
    model,
    RECIPE,
);
/// llama-server's normalized and raw vectors.
pub static SERVER: Family = family(
    "server-gemma-embedding",
    &["ref_gemma_embedding_server"],
    Kind::Server,
    &PROFILE,
    ARCH,
    model,
    RECIPE,
);
/// The architecture's families in reader order.
pub static FAMILIES: &[&Family] = &[&TEXT, &TEXT_CPU, &SERVER];
#[cfg(test)]
mod tests;
