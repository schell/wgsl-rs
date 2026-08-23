//! Deshadowing context.
//!
//! Used by the deshadowing pass to produce unique names for shadowed name
//! bindings.

use std::collections::{HashMap, HashSet};

/// A single scope.
///
/// `names` tracks every name declared in this scope. `renames` maps an
/// original (shadowed) name to its mangled replacement — only populated
/// for same-scope shadows. A fresh declaration goes into `names` but not
/// `renames`, which acts as a "stop" signal during resolution: an outer
/// scope's rename of the same name must not apply past this point.
#[derive(Default)]
struct DeshadowScope {
    names: HashSet<String>,
    renames: HashMap<String, String>,
}

/// Scope-tracking context for the deshadow pass.
///
/// Resolution of a reference walks the scope stack innermost-first
/// ([`resolve_rename`]). If a scope has an entry in `renames` for the
/// name, that rename applies. If a scope has the name in `names` but
/// not in `renames`, the name was freshly declared here, so any
/// outer-scope rename is suppressed (this is what prevents renames from
/// leaking past a nested-scope redeclaration). Otherwise the walk
/// continues to the next outer scope.
#[derive(Default)]
pub struct DeshadowCtx {
    /// Stack of scopes, innermost last.
    scopes: Vec<DeshadowScope>,
    /// Monotonic counter for generating unique names.
    counter: usize,
}

impl DeshadowCtx {
    /// Generate a unique name for a shadowed variable. Checks all scopes
    /// on the stack to avoid collisions with any in-scope name.
    fn unique_name(&mut self, original: &str) -> String {
        loop {
            self.counter += 1;
            let candidate = format!("{original}_{}", self.counter);
            if !self.scopes.iter().any(|s| s.names.contains(&candidate)) {
                return candidate;
            }
        }
    }

    /// Check if a name is already declared in the current (innermost)
    /// scope.
    fn current_scope_contains(&self, name: &str) -> bool {
        self.scopes.last().is_some_and(|s| s.names.contains(name))
    }

    /// Push a new scope onto the stack.
    pub fn push_scope(&mut self) {
        self.scopes.push(Default::default());
    }

    /// Pop the deepest scope off the stack.
    pub fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    /// Declare a name in the current scope. If the name shadows an
    /// existing same-scope binding, generates a unique mangled name,
    /// inserts *that* into the scope's `names` and records the rename in
    /// the scope's `renames`, and returns the new name. If the name is
    /// fresh, inserts it into `names` and returns `None`.
    pub fn declare_name(&mut self, original: &str) -> Option<String> {
        if self.scopes.is_empty() {
            self.push_scope();
        }
        let is_shadow = self.current_scope_contains(original);
        if is_shadow {
            let new_name = self.unique_name(original);
            let scope = self
                .scopes
                .last_mut()
                .expect("scope was just ensured non-empty");
            scope.names.insert(new_name.clone());
            scope.renames.insert(original.to_string(), new_name.clone());
            Some(new_name)
        } else {
            let scope = self
                .scopes
                .last_mut()
                .expect("scope was just ensured non-empty");
            scope.names.insert(original.to_string());
            None
        }
    }

    /// Resolve a reference to a name by walking the scope stack
    /// innermost-first. Returns the mangled name if an enclosing scope
    /// shadowed this name (and no intervening scope freshly declared
    /// it), or `None` if the name should be used as-is.
    pub fn resolve_rename(&self, name: &str) -> Option<&str> {
        for scope in self.scopes.iter().rev() {
            if let Some(new_name) = scope.renames.get(name) {
                return Some(new_name.as_str());
            }
            if scope.names.contains(name) {
                // Fresh declaration in this scope suppresses any
                // outer-scope rename.
                return None;
            }
        }
        None
    }
}
