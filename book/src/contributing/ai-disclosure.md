# AI Disclosure Policy

All AI-generated or AI-collaborated contributions to wgsl-rs must be disclosed. This policy follows [NLnet's Generative AI Disclosure Policy](https://nlnet.nl/foundation/policies/generativeAI/).

## Commit Author Format

Commits involving an LLM use a compound author string:

```
{human-author} with {llm-name} {llm-version} <{human-email}>
```

Example:

```
Schell Scivally with Claude Sonnet 4.5 <schell@example.com>
```

## Two-Step Process

Because `git commit` cannot set a custom author string directly in a single invocation, use two commands:

1. Create the commit normally:

   ```sh
   git commit -m "Add SlabItemExt slab_read codegen"
   ```

2. Amend the author:

   ```sh
   git commit --amend --author="Schell Scivally with Claude Sonnet 4.5 <schell@example.com>"
   ```

## When to Disclose

Disclose any contribution where an LLM authored or materially co-authored code, documentation, tests, or commit messages. Pure typo fixes suggested by a human reviewer do not require disclosure, but when in doubt, disclose.