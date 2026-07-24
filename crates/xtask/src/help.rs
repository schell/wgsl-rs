/// Command output.
pub struct Output {
    pub status: std::process::ExitStatus,
    pub stdout: String,
    pub stderr: String,
}

impl Output {
    pub fn ensure_success(&self, msg: &str) {
        if !self.status.success() {
            eprintln!("{msg}: \n{}\n{}", self.stdout, self.stderr);
            std::process::exit(1);
        }
    }
}

/// Run a shell command streaming stdout and stderr to the terminal, and exit(1)
/// if exit status is not success.
pub fn cmd_with_env(command: &str, args: impl AsRef<str>, env_var: Option<(&str, &str)>) {
    log::trace!("Running '{command}' without capture");
    let args = args.as_ref().split_ascii_whitespace().collect::<Vec<_>>();
    log::trace!("  args: {args:?}");
    let mut cmd = std::process::Command::new(command);
    cmd.args(args)
        .stdout(std::process::Stdio::inherit())
        .stderr(std::process::Stdio::inherit());
    if let Some((key, val)) = env_var {
        cmd.env(key, val);
    }
    let status = cmd.status().unwrap_or_else(|e| {
        eprintln!("Could not execute command '{command}': {e}");
        std::process::exit(1);
    });
    if !status.success() {
        eprintln!("Command '{command}' did not exit success: {status}");
        std::process::exit(1);
    }
}

/// Run a shell command streaming stdout and stderr to the terminal. Exits
/// the process with status 1 if the command fails (does not panic).
pub fn cmd(command: &str, args: impl AsRef<str>) {
    cmd_with_env(command, args, None)
}

/// Run a shell command and return its captured output without printing.
pub fn cmd_capture(command: &str, args: impl AsRef<str>) -> Output {
    log::trace!("Running '{command}'");
    let args = args.as_ref().split_ascii_whitespace().collect::<Vec<_>>();
    log::trace!("  args: {args:?}");
    let mut cmd = std::process::Command::new(command);
    cmd.args(args);
    match cmd.output() {
        Ok(std::process::Output {
            status,
            stdout,
            stderr,
        }) => Output {
            status,
            stdout: String::from_utf8_lossy(&stdout).into_owned(),
            stderr: String::from_utf8_lossy(&stderr).into_owned(),
        },
        Err(e) => {
            eprintln!("Could not execute command '{command}': {e}");
            std::process::exit(1);
        }
    }
}

/// Returns whether the given binary exists and is reachable on PATH.
pub fn does_binary_exist(name: &str) -> bool {
    log::trace!("Does binary '{name}' exist?");
    let exists = std::process::Command::new(name)
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    log::trace!("  {exists}");
    exists
}
