//! Subcommand that performs CI actions.

use crate::help::*;

#[derive(clap::Subcommand)]
pub enum CiAction {
    /// Run unit tests with `nextest`.
    UnitTests,
    /// Run roundtrip tests.
    RoundtripTests,
    /// Run documentation tests.
    DocTests,
    /// Run all tests.
    AllTests,
    /// Install clippy CI deps.
    InstallClippyDeps,
    /// Run clippy lints.
    Clippy,
    /// Run cargo fmt.
    Fmt,
    /// Run cargo doc generation and err on any warnings.
    Docs,
    /// Run a pull-request check, including:
    /// * clippy lints
    /// * formatting
    /// * tests
    /// * documentation
    PrCheck,
}

impl CiAction {
    pub fn ensure_cargo_nextest() {
        if !does_binary_exist("cargo-nextest") {
            log::info!("Installing cargo-nextest...");
            cmd_capture("cargo", "install --locked cargo-nextest")
                .ensure_success("Could not install cargo-nextest");
            log::info!("...done!");
        }
    }

    pub fn ensure_clippy_sarif() {
        if !does_binary_exist("clippy-sarif") {
            log::info!("Installing clippy-sarif");
            cmd_capture("cargo", "install clippy-sarif")
                .ensure_success("Could not install clippy-sarif");
        }
        if !does_binary_exist("sarif-fmt") {
            log::info!("Installing sarif-fmt");
            cmd_capture("cargo", "install sarif-fmt").ensure_success("Could not install sarif-fmt");
        }
    }

    pub fn run(&self) {
        match self {
            CiAction::UnitTests => {
                Self::ensure_cargo_nextest();
                cmd("cargo", "nextest run --all-features");
            }
            CiAction::RoundtripTests => {
                cmd("cargo", "run -p roundtrip-tests -- --list");
                cmd("cargo", "run -p roundtrip-tests");
            }
            CiAction::DocTests => cmd("cargo", "test --doc --all-features"),
            CiAction::AllTests => {
                CiAction::UnitTests.run();
                CiAction::RoundtripTests.run();
                CiAction::DocTests.run();
            }
            CiAction::InstallClippyDeps => Self::ensure_clippy_sarif(),
            CiAction::Clippy => cmd(
                "cargo",
                "clippy --all-features --all-targets -- -D warnings",
            ),

            CiAction::Fmt => cmd("cargo", "fmt --all -- --check"),
            CiAction::Docs => cmd_with_env(
                "cargo",
                "doc --all-features --no-deps",
                Some(("RUSTDOCFLAGS", "-D warnings")),
            ),
            CiAction::PrCheck => {
                CiAction::Fmt.run();
                CiAction::Clippy.run();
                CiAction::AllTests.run();
            }
        }
    }
}
