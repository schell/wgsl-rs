//! This program is a development tool to be used by agents, human or otherwise.

use clap::Parser;

mod wgsl_spec;

#[derive(clap::Parser)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(clap::Subcommand)]
enum Commands {
    /// Fetch information from the WGSL specification
    WgslSpec {
        #[command(subcommand)]
        action: wgsl_spec::WgslSpecAction,
    },
}

fn main() {
    env_logger::builder().init();
    let cli = Cli::parse();

    match cli.command {
        Commands::WgslSpec { action } => action.run(),
    }
}
