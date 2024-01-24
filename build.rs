use std::{env, path::PathBuf};

use cmake::Config;

fn main() {
    // Build hptt
    let extern_path = PathBuf::from("extern");
    let hptt_path = extern_path.join("hptt");
    let dst = Config::new(hptt_path).build();

    // Link it
    println!("cargo:rustc-link-search={}", dst.join("lib").display());
    println!("cargo:rustc-link-lib=hptt");
    println!("cargo:rustc-link-lib=stdc++");

    // Generate bindings
    let header = extern_path.join("api.h");
    let bindings = bindgen::Builder::default()
        .header(header.to_str().unwrap())
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings");
}
