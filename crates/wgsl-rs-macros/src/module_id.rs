//! Assignment of unique ids to `#[wgsl]` modules.
//!
//! Every expanded module gets a `u64` id baked into its `WGSL_SOURCE`
//! static. The runtime deduplicates imported sources by id (diamond
//! import graphs), so two *distinct* modules must never share an id:
//! one would be silently dropped from the assembled WGSL, and the
//! shader would fail to parse with "unknown identifier" errors.
//!
//! A plain per-process counter is not enough. The proc-macro's statics
//! are per rustc invocation, so the counter restarts at 0 for every
//! crate — and every target — being compiled. Two modules in different
//! crates routinely shared an id (wgsl-rs#165), the assembler dropped
//! one as a "duplicate" import, and the surviving module's calls into
//! the dropped one failed with unknown-identifier parse errors.
//!
//! Instead the id mixes a stable hash of the *consuming compilation
//! unit's* identity with the per-process counter:
//!
//! * **Within one compilation unit**, only the counter varies, and XOR with a
//!   fixed hash is a bijection in the counter — ids are *exactly* unique,
//!   matching the old scheme's guarantee.
//! * **Across compilation units**, ids are separated by hashing each unit's
//!   identity: the Cargo package (`CARGO_PKG_NAME` + `CARGO_PKG_VERSION`) plus
//!   the target being compiled (`CARGO_CRATE_NAME`, and `CARGO_BIN_NAME` for
//!   binary targets, so a package's lib and its same-named bin don't alias).
//!   Including the version also separates two semver-incompatible copies of one
//!   package, which cargo can legally compile into one binary.
//!
//! This cross-unit separation is *probabilistic*, like any hash: it
//! fails only if FNV-1a maps two distinct identities to hashes whose
//! difference equals a small counter XOR. For 64-bit FNV-1a over
//! realistic name/version strings the collision probability per pair
//! is about 2^-64, and the birthday bound only becomes relevant
//! around 2^32 modules. That residual risk is accepted: exact ids
//! would require widening the public `u64` id to a string or
//! (crate, path) key.
//!
//! Known residual edge: two targets of one package whose
//! `CARGO_CRATE_NAME` values alias (e.g. an integration test named
//! exactly after the lib crate) still share an identity; stable cargo
//! provides no target-kind environment variable to discriminate them.
//!
//! Outside cargo these environment variables are absent and ids
//! degrade to the old counter-only scheme, so non-cargo builds behave
//! exactly as before.

use std::sync::atomic::{AtomicU64, Ordering};

/// Per-process count of `#[wgsl]` modules expanded so far. Combined
/// with the consuming compilation unit's identity hash in
/// [`next_module_id`] so ids are unique within a unit and, with
/// overwhelming probability, distinct across units.
static NEXT_MODULE_ID: AtomicU64 = AtomicU64::new(0);

/// FNV-1a 64-bit hash, streamed over the given byte slices.
///
/// Chosen because it is small and *stable*: ids are baked into each
/// crate at its own compile time, so the hash must not vary between
/// rustc versions (`DefaultHasher` gives no such guarantee).
fn fnv1a64(slices: &[&[u8]]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &slice in slices {
        for &byte in slice {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    hash
}

/// Derive the module id for a module expanded in the compilation unit
/// identified by `pkg_name`/`pkg_version`/`crate_name`/`bin_name`, at
/// the given per-process `counter` position.
///
/// Pure, so the collision properties can be unit-tested directly;
/// [`next_module_id`] reads the environment and counter and delegates
/// here.
fn module_id_for(
    pkg_name: &str,
    pkg_version: &str,
    crate_name: &str,
    bin_name: &str,
    counter: u64,
) -> u64 {
    // NUL separators keep identity tuples like ("ab", "1.0", ...) and
    // ("ab1", ".0", ...) from hashing to the same stream.
    fnv1a64(&[
        pkg_name.as_bytes(),
        b"\0",
        pkg_version.as_bytes(),
        b"\0",
        crate_name.as_bytes(),
        b"\0",
        bin_name.as_bytes(),
    ]) ^ counter
}

/// Allocate the id for the next `#[wgsl]` module expanded in this
/// process.
///
/// Reads the consuming compilation unit's identity from the
/// cargo-provided environment (`CARGO_PKG_NAME`, `CARGO_PKG_VERSION`,
/// `CARGO_CRATE_NAME`, `CARGO_BIN_NAME`). Outside cargo these are
/// absent and ids degrade to the counter-only scheme: still unique
/// within a single compilation, but with no cross-unit separation
/// (exactly the old behavior).
pub(crate) fn next_module_id() -> u64 {
    let pkg_name = std::env::var("CARGO_PKG_NAME").unwrap_or_default();
    let pkg_version = std::env::var("CARGO_PKG_VERSION").unwrap_or_default();
    let crate_name = std::env::var("CARGO_CRATE_NAME").unwrap_or_default();
    let bin_name = std::env::var("CARGO_BIN_NAME").unwrap_or_default();
    let counter = NEXT_MODULE_ID.fetch_add(1, Ordering::Relaxed);
    module_id_for(&pkg_name, &pkg_version, &crate_name, &bin_name, counter)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FNV-1a is pinned by reference vectors so an accidental algorithm
    /// change — which would silently re-assign every module id — can't
    /// slip through unnoticed.
    #[test]
    fn fnv1a64_matches_reference_vectors() {
        assert_eq!(fnv1a64(&[b""]), 0xcbf29ce484222325);
        assert_eq!(fnv1a64(&[b"a"]), 0xaf63dc4c8601ec8c);
        assert_eq!(fnv1a64(&[b"foobar"]), 0x85944171f73967e8);
    }

    /// Two modules in the *same* compilation unit (different counters)
    /// must get different ids — exactly, not just with high
    /// probability: XOR with a fixed hash is a bijection in the
    /// counter.
    #[test]
    fn ids_differ_within_a_compilation_unit() {
        let ids: Vec<u64> = (0..8)
            .map(|i| module_id_for("my-game", "0.1.0", "my_game", "", i))
            .collect();
        for (i, &a) in ids.iter().enumerate() {
            for &b in &ids[i + 1..] {
                assert_ne!(a, b, "same unit, different counters must not collide");
            }
        }
    }

    /// The bug from wgsl-rs#165: module 0 of crate `shader-a` and
    /// module 0 of crate `shader-b` used to *both* get id 0, so one
    /// import was dropped as a duplicate. Package identity must keep
    /// them apart.
    #[test]
    fn ids_differ_across_crates() {
        let a = module_id_for("shader-a", "0.1.0", "shader_a", "", 0);
        let b = module_id_for("shader-b", "0.1.0", "shader_b", "", 0);
        assert_ne!(a, b, "first modules of different crates must not collide");
    }

    /// The same-package/different-target hole called out in review of
    /// PR #173: a package's lib target and one of its integration-test
    /// or bin targets share `CARGO_PKG_NAME`/`CARGO_PKG_VERSION`, but
    /// each starts its own proc-macro counter at 0. The crate/bin name
    /// discriminators must keep their ids apart.
    #[test]
    fn ids_differ_across_targets_of_same_package() {
        // Lib target of package `id-collision-app`: crate name matches
        // the package, no bin involved.
        let lib = module_id_for("id-collision-app", "0.1.0", "id_collision_app", "", 0);
        // An integration-test target compiled from
        // tests/target_collision.rs: crate name is the test's stem.
        let test = module_id_for("id-collision-app", "0.1.0", "target_collision", "", 0);
        // The package's own bin: `CARGO_CRATE_NAME` equals the lib's,
        // so `CARGO_BIN_NAME` is what distinguishes it.
        let bin = module_id_for(
            "id-collision-app",
            "0.1.0",
            "id_collision_app",
            "id-collision-app",
            0,
        );
        assert_ne!(
            lib, test,
            "lib and integration-test targets must not collide"
        );
        assert_ne!(lib, bin, "lib and bin targets must not collide");
        assert_ne!(test, bin, "test and bin targets must not collide");
    }

    /// Sweep a realistic id space — the first 16 modules of each of
    /// three crates, pairwise — so no sampled (crate, counter)
    /// combination coincides. Cross-identity uniqueness is inherently
    /// probabilistic; this pins it for the sampled space.
    #[test]
    fn ids_differ_across_crates_for_many_counters() {
        for a in 0..16u64 {
            for b in 0..16u64 {
                for (left, right) in [
                    ("shader-a", "shader-b"),
                    ("shader-a", "shader-c"),
                    ("shader-b", "shader-c"),
                ] {
                    assert_ne!(
                        module_id_for(left, "0.1.0", left, "", a),
                        module_id_for(right, "0.1.0", right, "", b),
                        "{left} #{a} vs {right} #{b}"
                    );
                }
            }
        }
    }

    /// Two semver-incompatible versions of the same package can coexist
    /// in one build graph; the version must keep their ids apart.
    #[test]
    fn ids_differ_across_versions_of_same_crate() {
        let old = module_id_for("shader-lib", "1.0.0", "shader_lib", "", 0);
        let new = module_id_for("shader-lib", "2.0.0", "shader_lib", "", 0);
        assert_ne!(
            old, new,
            "same crate name, different versions must not collide"
        );
    }

    /// Identity tuples whose concatenated byte streams would alias —
    /// ("ab", "1.0", ...) vs ("ab1", ".0", ...) — must hash
    /// differently; the NUL separators keep them distinct.
    #[test]
    fn identity_tuples_dont_alias() {
        assert_ne!(
            module_id_for("ab", "1.0", "ab", "", 0),
            module_id_for("ab1", ".0", "ab1", "", 0)
        );
    }
}
