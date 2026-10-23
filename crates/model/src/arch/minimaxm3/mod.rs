//! `minimax-m3` — MiniMax-M3. The file's keys (`hparams`), every tensor's role (`roles`), the
//! typed description (`spec`), the tensor names a step reads (`names`), the file planned onto a
//! machine (`place`) and the host tier's view of a routed layer (`host`).
//!
//! Every layer is a GQA layer with a per-head q/k norm and a partial NeoX rope. Attention and the
//! feed-forward block are separate facts. Absent `attention.sparse_layers`, the first
//! `leading_dense_block_count` layers attend every position and run a dense `swigluoai` block, and
//! the rest attend the blocks their indexer picks (MiniMax Sparse Attention) and route by a
//! sigmoid noaux_tc router over `swigluoai` experts beside an ungated shared one. Present, the
//! array names the sparse-attention layers and the dense lead still names the dense blocks. The
//! dialect's authority is llama.cpp's `src/models/minimax-m3.cpp`.

pub mod host;
pub mod hparams;
pub mod names;
pub mod place;
pub mod roles;
pub mod spec;
