//! GPU vs CPU roundtrip tests for `wgsl_rs::std` builtin functions.
//!
//! This binary validates the "two worlds" coherence of `wgsl-rs` by running
//! compute shaders on both the GPU (via wgpu) and the CPU (via
//! `dispatch_workgroups`), then comparing results within tolerance.
//!
//! Each test category defines one or more `#[wgsl]` compute shader modules
//! that exercise a family of builtin functions. The harness uploads input data
//! to a storage buffer, dispatches on both GPU and CPU, reads back results,
//! and reports any discrepancies.
//!
//! ## Usage
//!
//! ```text
//! cargo run -p roundtrip-tests                 # Run all tests
//! cargo run -p roundtrip-tests -- --list       # List available test categories
//! cargo run -p roundtrip-tests -- --filter trig  # Run only tests matching "trig"
//! ```

mod harness;
mod shaders;

use clap::Parser;

/// GPU vs CPU roundtrip tests for wgsl_rs::std builtin functions.
#[derive(Parser)]
#[command(name = "roundtrip-tests")]
struct Cli {
    /// List available test categories without running them.
    #[arg(long)]
    list: bool,

    /// Run only tests whose name contains this substring.
    #[arg(long)]
    filter: Option<String>,
}

fn main() {
    let cli = Cli::parse();
    let all_tests = shaders::all_tests();

    if cli.list {
        eprintln!("[roundtrip-tests] Available test categories:");
        for test in &all_tests {
            eprintln!("  {:<16} {}", test.name(), test.description());
        }
        return;
    }

    let tests: Vec<_> = match &cli.filter {
        Some(filter) => all_tests
            .into_iter()
            .filter(|t| t.name().contains(filter.as_str()))
            .collect(),
        None => all_tests,
    };

    if tests.is_empty() {
        eprintln!("[roundtrip-tests] No tests matched the filter.");
        std::process::exit(1);
    }

    let Some((device, queue)) = harness::create_device() else {
        eprintln!("[roundtrip-tests] No GPU adapter found. Cannot run roundtrip tests.");
        std::process::exit(1);
    };

    let mut total_pass = 0;
    let mut total_fail = 0;

    for test in &tests {
        let comparison_results = test.run(&device, &queue);

        for result in &comparison_results {
            if result.passed {
                total_pass += 1;
                eprintln!(
                    "[PASS] {} — max error: {:.2e}",
                    result.name, result.max_error,
                );
            } else {
                total_fail += 1;
                eprintln!(
                    "[FAIL] {} — {} mismatches:",
                    result.name,
                    result.mismatches.len()
                );
                for mismatch in &result.mismatches {
                    eprintln!("{mismatch}");
                }
            }
        }
    }

    eprintln!();
    eprintln!(
        "[roundtrip-tests] {total_pass} passed, {total_fail} failed out of {} sub-tests",
        total_pass + total_fail,
    );

    if total_fail > 0 {
        std::process::exit(1);
    }
}
