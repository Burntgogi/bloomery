//! The encoder's bytes on its card, as every projector type states them: what a plan reserves
//! for the tower and what the loaded tower holds itself to.
//!
//! A projector module computes its own figure from its hyperparameters (`deepseek41v::card::of`,
//! `qwen3vl::card::of`); the figure's shape is the same for all of them.

/// The encoder's bytes on its card for a file of given hyperparameters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CardBytes {
    /// The weights the encoder uploads.
    pub weights: u64,
    /// The activations it allocates.
    pub scratch: u64,
}

impl CardBytes {
    /// The figure a plan reserves for the encoder.
    #[must_use]
    pub const fn total(self) -> u64 {
        self.weights + self.scratch
    }
}

/// The bytes of an encoder's activation buffers: its RoPE table holds `f32_values` values at four
/// bytes each, every other buffer's `bf16_values` are two bytes each.
#[must_use]
pub(crate) fn activation_bytes(
    f32_values: usize,
    bf16_values: impl IntoIterator<Item = usize>,
) -> u64 {
    4 * f32_values as u64 + 2 * bf16_values.into_iter().map(|v| v as u64).sum::<u64>()
}
