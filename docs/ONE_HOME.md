# One home for Anchor

Until 2026-09-29 Anchor's code lived in two places: this repository, and a copy inside Vitna's private repository. The copy was an `engine/` directory and sixteen JavaScript modules under `platform/runtime/`. Two copies of an engine drift apart, and these had. This file records which copy of each piece is canonical, why, and what happens to the other.

**Compared.** This repository at `429581a` (its `main` since 2026-09-16) against Vitna's `main` at `9f72a77` (2026-09-29). Files were compared as git blobs, so line endings in a checkout do not count as drift. Vitna's `engine/` and `platform/runtime/` had not changed between that commit and `f5108ef`, where the comparison was first run.

## The decision

This repository is the only home for the engine and for Anchor's own modules. The licence, the releases and the issue tracker are here, and it is public, so anyone can check what the engine does. Vitna keeps no copy of Anchor's code. Once gate A3 passes, Vitna reaches Anchor the way it reaches any other local server: over HTTP, through its runtime's local route, at an address a policy names.

## The engine

`engine/src` and `engine/include` are byte-identical in the two copies, and Vitna's copy has no `CMakeLists.txt`. This copy is canonical, and gate A0 ([#2](https://github.com/convexityos/vitna-anchor/pull/2)) has since changed it. Vitna deletes its `engine/` directory.

## The sixteen runtime modules

Whose a module is was decided by what imports it. Vitna's gateway server imports `cache`, `compaction`, `guard`, `ledger` and `queue`. In this repository only the invented server behind `vitna-anchor serve` used them, and A0 removed that server. The other eleven exist only for Anchor.

| Module | Whose | The two copies | Canonical | Here |
|---|---|---|---|---|
| `anchor-run.mjs` | Anchor | Differ. Vitna's drops unused imports and comments two empty `catch` blocks | This one, as A0 rewrote it | Kept |
| `anchor-chat.mjs` | Anchor | Differ. Vitna's drops an unused helper | Neither | Deleted by A0. It only talked to the invented server |
| `grammar.mjs` | Anchor | Differ, in lint only: a comment on a variable nothing reads, a comment in an empty `catch`, a needless escape | This one, with those edits ported | Kept |
| `ingest.mjs` | Anchor | Differ. One regex escape, with the same meaning | This one, with the edit ported | Kept |
| `quantize.mjs` | Anchor | Differ. Vitna's drops two unused imports and an unused `statSync` call | This one, with the edits ported | Kept |
| `stripe.mjs` | Anchor | Differ. Comments in two empty `catch` blocks | This one, as A0 rewrote it, with the comments ported | Kept |
| `tune.mjs` | Anchor | Differ. One unused import | Neither | Deleted by A0. Its recommendations came from reading back a file it had just written |
| `arbitrage.mjs` | Anchor | Identical | Neither | Deleted by A0. Its price table had no source |
| `draft.mjs` | Anchor | Identical | Neither | Deleted by A0. It simulated speculative decoding |
| `prefetch.mjs` | Anchor | Identical | Neither | Deleted by A0. It simulated prefetch |
| `registry.mjs` | Anchor | Identical | Neither | Deleted by A0. It listed models as verified that nothing had verified |
| `cache.mjs` | Vitna's gateway | Identical | Vitna's | Deleted here |
| `guard.mjs` | Vitna's gateway | Identical | Vitna's | Deleted here |
| `queue.mjs` | Vitna's gateway | Identical | Vitna's | Deleted here |
| `compaction.mjs` | Vitna's gateway | Differ. Vitna's keeps `additionalProperties: false` in a tool schema, which this copy deleted as padding. It is a constraint, and a provider's strict mode can require it | Vitna's, which has that fix | Deleted here |
| `ledger.mjs` | Vitna's gateway | Differ. Vitna's no longer fills in a hardware cost, a cloud price, a saving, `airgap: true` and a precision tier for rows the local route served, none of which anything measured. It also records a request id and two price fields | Vitna's, which has those fixes | Deleted here |

For the six Anchor modules that survive or differ, this copy is canonical even though Vitna's edits came later, because those edits change no behaviour. They are ported here, so nothing is lost when Vitna deletes its copies. For the two gateway modules that differ, Vitna's later changes are fixes to its own runtime, and this repository had no reason to keep either module.

## Vitna's side

Vitna's side is a pull request in Vitna's repository. It deletes `engine/` and the eleven Anchor modules, with the tests and package scripts that run them. The /anchor page then takes its status from this repository rather than from a local copy of the code. The note for that pull request went to the owner, since it describes a private repository. This one records only the decision.
