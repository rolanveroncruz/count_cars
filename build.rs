use std::env;
use std::path::PathBuf;

fn main() {
    // 1. Tell Cargo to link your Rust binary against the libhailort.so library
    println!("cargo:rustc-link-lib=hailort");

    // 2. Configure bindgen to read the exact header installed on your Pi
    let bindings = bindgen::Builder::default()
        .header("/usr/include/hailo/hailort.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .prepend_enum_name(false)
        .generate()
        .expect("Unable to generate HailoRT bindings from system headers");

    // 3. Write the generated Rust code to the output directory
    let _ = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file("src/inference/bindings.rs")
        .expect("Couldn't write bindings!");
}