use std::{env, path::PathBuf};

use bindgen::callbacks::ParseCallbacks;
use cmake::Config;

#[derive(Debug)]
struct DoxygenCallback;

impl ParseCallbacks for DoxygenCallback {
    fn process_comment(&self, comment: &str) -> Option<String> {
        let transformed = doxygen_rs::generator::rustdoc(comment.into());
        Some(transformed.unwrap_or_else(|_| comment.into()))
    }
}

fn main() {
    // Build hptt
    let extern_path = PathBuf::from("extern");
    let hptt_path = extern_path.join("hptt");
    let dst = Config::new(hptt_path).build();

    // Link it
    println!("cargo:rustc-link-search={}", dst.join("lib").display());
    println!("cargo:rustc-link-lib=hptt");

    // Link the C++ standard library
    if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    } else if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=dylib=c++");
    }

    // Generate bindings
    let header = extern_path.join("api.h");
    let bindings = bindgen::Builder::default()
        .header(header.to_str().unwrap())
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .parse_callbacks(Box::new(DoxygenCallback))
        .generate()
        .expect("Unable to generate bindings");
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings");
}
