fn main() {
    let output = std::process::Command::new("rustc")
        .arg("--version")
        .output()
        .expect("Failed to run rustc --version");

    let version = String::from_utf8_lossy(&output.stdout);

    if version.contains("nightly") {
        println!("cargo:rustc-cfg=nightly");
    }

    // Tell cargo about our custom cfg to avoid unexpected_cfgs warnings
    println!("cargo::rustc-check-cfg=cfg(nightly)");

    println!("cargo:rerun-if-env-changed=RUSTC");
}
