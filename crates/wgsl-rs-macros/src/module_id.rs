//! Assignment of unique ids to `#[wgsl]` modules.
//!
//! Every expanded module gets a `u64` id baked into its `WGSL_SOURCE`
//! static. The runtime deduplicates imported sources by id (diamond
//! import graphs), so two *distinct* modules must never share an id:
//! one would be silently dropped from the assembled WGSL, and the
//! shader would fail to parse with "unknown identifier" errors.
//!
//! A plain per-process counter is not enough. The proc-macro's statics
//! are per compiled crate, so the counter restarts at 0 for every crate
//! in a workspace — modules in different crates collided (wgsl-rs#165).
//!
//! Instead the id mixes a stable hash of the *consuming crate's*
//! identity (`CARGO_PKG_NAME` + `CARGO_PKG_VERSION`, which cargo sets
//! for the crate being compiled) with the per-process counter:
//!
//! * within one crate, only the counter varies, keeping ids unique — the same
//!   guarantee the old scheme provided;
//! * across crates, the crate-identity hash keeps ids unique.
//!
//! The version participates so that two semver-incompatible versions
//! of the same package — which cargo happily compiles into one binary
//! — don't collide with each other.

use std::sync::atomic::{AtomicU64, Ordering};

/// Per-process count of `#[wgsl]` modules expanded so far. Combined
/// with the consuming crate's identity hash in [`next_module_id`] so ids
/// are unique both within a crate (the counter) and across crates (the
/// hash).
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

/// Derive the module id for a module expanded in the crate identified by
/// `crate_name`/`crate_version`, at the given per-process `counter`
/// position.
///
/// Pure, so the collision guarantees can be unit-tested directly;
/// [`next_module_id`] reads the environment and counter and delegates
/// here.
fn module_id_for(crate_name: &str, crate_version: &str, counter: u64) -> u64 {
    // The NUL separator keeps identity pairs like ("ab", "1.0") and
    // ("ab1", ".0") from hashing to the same stream.
    fnv1a64(&[crate_name.as_bytes(), b"\0", crate_version.as_bytes()]) ^ counter
}

/// Allocate the id for the next `#[wgsl]` module expanded in this
/// process.
///
/// Reads the consuming crate's identity from the cargo-provided
/// environment (`CARGO_PKG_NAME`, `CARGO_PKG_VERSION`). Outside cargo
/// these are absent and ids degrade to the counter-only scheme: still
/// unique within a single compilation, but with no cross-crate
/// guarantee (exactly the old behavior).
pub(crate) fn next_module_id() -> u64 {
    let crate_name = std::env::var("CARGO_PKG_NAME").unwrap_or_default();
    let crate_version = std::env::var("CARGO_PKG_VERSION").unwrap_or_default();
    let counter = NEXT_MODULE_ID.fetch_add(1, Ordering::Relaxed);
    module_id_for(&crate_name, &crate_version, counter)
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

    /// Two modules in the *same* crate (different counters) must get
    /// different ids — the pre-existing single-crate guarantee. XOR with
    /// a fixed hash is a bijection in the counter, so this must hold
    /// exactly, not just with high probability.
    #[test]
    fn ids_differ_within_a_crate() {
        let ids: Vec<u64> = (0..8)
            .map(|i| module_id_for("my-game", "0.1.0", i))
            .collect();
        for (i, &a) in ids.iter().enumerate() {
            for &b in &ids[i + 1..] {
                assert_ne!(a, b, "same crate, different counters must not collide");
            }
        }
    }

    /// The bug from wgsl-rs#165: module 0 of crate `shader-a` and module
    /// 0 of crate `shader-b` used to *both* get id 0, so one import was
    /// dropped as a duplicate. Crate identity must keep them apart.
    #[test]
    fn ids_differ_across_crates() {
        let a = module_id_for("shader-a", "0.1.0", 0);
        let b = module_id_for("shader-b", "0.1.0", 0);
        assert_ne!(a, b, "first modules of different crates must not collide");
    }

    /// Sweep a realistic id space — the first 16 modules of each of
    /// three crates, pairwise — so no (crate, counter) combination
    /// coincides.
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
                        module_id_for(left, "0.1.0", a),
                        module_id_for(right, "0.1.0", b),
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
        let old = module_id_for("shader-lib", "1.0.0", 0);
        let new = module_id_for("shader-lib", "2.0.0", 0);
        assert_ne!(
            old, new,
            "same crate name, different versions must not collide"
        );
    }

    /// ("ab", "1.0") and ("ab1", ".0") alias to the same concatenated
    /// byte stream; the NUL separator keeps their ids distinct.
    #[test]
    fn crate_name_version_pairs_dont_alias() {
        assert_ne!(module_id_for("ab", "1.0", 0), module_id_for("ab1", ".0", 0));
    }
}
