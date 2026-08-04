# xtask & CI

wgsl-rs uses a `cargo xtask` workflow for repository maintenance tasks and a GitHub Actions CI pipeline for pull-request validation.

## xtask Commands

### `wgsl-spec`

Fetch and process the WGSL specification, useful for regenerating reference tables and validation data.

```sh
cargo xtask wgsl-spec toc/section
```

### `ci`

Run the same checks that CI enforces, locally:

```sh
cargo xtask ci pr-check
```

Always run `pr-check` before pushing a pull request.

## CI Workflow

The CI pipeline runs the following jobs:

| Job     | Tool          | Notes                                                    |
|---------|---------------|----------------------------------------------------------|
| fmt     | `cargo +nightly fmt --check` | Formatting requires the nightly toolchain  |
| clippy  | `cargo clippy`                | All warnings must be clean                 |
| test    | `cargo test`                  | Runs on macOS so GPU tests execute          |
| docs    | `cargo doc`                   | Documentation must build without warnings  |

### GPU Tests on macOS

The `test` job runs on macOS because the project's GPU integration tests require a working GPU context. Ensure local `pr-check` runs pass on macOS before pushing if you have modified anything under `linkage` or validation.

## Before Pushing a PR

1. Run `cargo xtask ci pr-check`.
2. Fix any fmt, clippy, test, or doc failures.
3. Push and open the PR with a description of the change.