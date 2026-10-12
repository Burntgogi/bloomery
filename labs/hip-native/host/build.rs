fn main() {
    println!("cargo:rerun-if-env-changed=HIP_LIB_DIR");
    let dir =
        std::env::var("HIP_LIB_DIR").expect("HIP_LIB_DIR names the HIP runtime library directory");
    println!("cargo:rustc-link-search=native={dir}");
    println!("cargo:rustc-link-lib=dylib=amdhip64");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{dir}");
}
