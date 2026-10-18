fn main() {
    let manifest = std::path::PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let fit = manifest.join("../../../crates/gpu/src/model/fit.rs");
    println!("cargo:rerun-if-changed={}", fit.display());
    let source = std::fs::read_to_string(fit).expect("read common load reserve");
    let reserve = source
        .lines()
        .find(|line| line.starts_with("pub const FIT_RESERVE:"))
        .expect("common load reserve declaration");
    let out = std::path::PathBuf::from(std::env::var_os("OUT_DIR").unwrap());
    std::fs::write(out.join("fit_reserve.rs"), reserve).expect("extract common load reserve");
    println!("cargo:rerun-if-env-changed=HIP_LIB_DIR");
    let dir =
        std::env::var("HIP_LIB_DIR").expect("HIP_LIB_DIR names the HIP runtime library directory");
    println!("cargo:rustc-link-search=native={dir}");
    println!("cargo:rustc-link-lib=dylib=amdhip64");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{dir}");
}
