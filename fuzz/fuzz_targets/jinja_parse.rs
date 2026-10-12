#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|source: &str| {
    let _ = jinja::ChatTemplate::parse(source);
});
