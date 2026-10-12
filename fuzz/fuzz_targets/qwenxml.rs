#![no_main]

use bloomery_fuzz::{Format, scan};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|text: &str| scan(Format::Qwen, text));
