# Architecture Decision Records

## 2026-08-18: Use a fixed, revocable machine credential for Raphael Workspace reads

Status: Accepted

Context:
Raphael Workspace needs server-to-server access to a small part of the Kanban
dashboard API. Reusing the browser's process-wide session bearer would give the
Workspace every dashboard capability, could not be revoked independently, and
would couple a service deployment to an interactive login. Creating a second
Kanban API or database would duplicate authority and eventually drift from the
native dashboard.

Decision:
- Keep the existing Kanban handlers as the only data authority. Nine existing
  `GET` routes opt into the shared dashboard token middleware while retaining
  their unchanged cookie/session behavior for interactive callers.
- Reserve the `hrw1_` credential family for this contour. A token is bound to
  the fixed `raphael-workspace` principal, `kanban.read` scope, project, board,
  and grant version. Any other path, method, method-override header, query
  shape, or cross-board object fails closed.
- Store only token identifiers and SHA-256 digests under `HERMES_HOME`; bearer
  plaintext is written once to an explicit owner-only file. Tokens have bounded
  expiry, support independent rotation and immediate revocation, and are
  verified from disk on every request.
- Read project and board state through SQLite `mode=ro` plus `query_only`.
  Machine responses expose only the fields consumed by Workspace, exclude
  recommendation cards from ordinary work, and serve attachment bytes only
  after board/task/path/type/size ownership checks.
- Durably audit issuance, revocation, allowed reads, and denials using the
  transport peer address. If the audit record cannot be written, the machine
  operation returns a generic `503` instead of proceeding unaudited.

Operations:
```bash
hermes kanban-workspace-token issue --out /owner-only/path/workspace.token --ttl-days 90
hermes kanban-workspace-token list
hermes kanban-workspace-token revoke <token_id>
```

For rotation, issue with `--replaces <active_token_id>`, deploy and verify the
new credential, then revoke the old identifier. Issuance never revokes the old
token automatically, so a failed cutover remains recoverable. This credential
is not a general public API key and grants no write, recommendation-decision,
profile-management, connector, pipeline, or deployment authority.

## 2026-07-13: Scope plugin manager state by Hermes home/profile (keyed cache)

Status: Accepted

Context:
Hermes supports multiple profiles via different Hermes home directories.
Homes are switched two ways in a running process: the `HERMES_HOME`
environment variable (single-profile CLI/gateway processes), and the
context-local `set_hermes_home_override()` (`hermes_constants.py`), which
the multiplexed gateway worker (`gateway/run.py`'s `_profile_scope`) and
subagent/embedded callers use to serve several profiles from one
long-lived process. The override is a `ContextVar` and deliberately does
**not** mutate `os.environ`, since that would leak one profile's home
into every other concurrent task in the same process.

The plugin manager was a process-global single-slot singleton
(`_plugin_manager`). User-installed plugins are discovered from
`get_hermes_home() / "plugins"`, and context-engine plugins (e.g.
`hermes-lcm`) capture profile-scoped state — such as the LCM database
path — at registration time. A single-slot cache meant:

1. Switching homes via `set_hermes_home_override()` was invisible to a
   naive "did `HERMES_HOME` change" check, so the singleton silently kept
   serving the first profile's manager to every other profile in the
   process.
2. Even when a fresh `PluginManager` *was* created for a new home, plugin
   modules are imported into `sys.modules` as `hermes_plugins.<slug>` by
   `_load_directory_module`, and only that top-level module was ever
   replaced. A same-slug plugin's *relative* imports
   (`from . import state`) are cached separately under
   `hermes_plugins.<slug>.<submodule>`, and Python's import machinery
   resolves those from `sys.modules` first — so a profile switch could
   silently keep serving a previous profile's already-imported submodule
   code/state instead of re-executing the new profile's plugin.

Decision:
- Replace the single-slot singleton with a cache keyed on the *resolved*
  Hermes home path (`_plugin_managers_by_home: Dict[Path, PluginManager]`).
  `get_plugin_manager()` resolves the current home via `get_hermes_home()`
  (which itself already consults `get_hermes_home_override()` before
  `os.environ`), so both the env-var and context-local override paths are
  covered uniformly.
- `_plugin_manager` (the old single-slot name) is kept as a thin "last
  manager returned" pointer purely for backward compatibility with
  existing test code that does
  `monkeypatch.setattr(plugins_mod, "_plugin_manager", some_manager)`.
  When that name is monkeypatched to a manager the keyed cache doesn't
  know about, `get_plugin_manager()` treats it as an explicit injection
  and adopts it into the cache under the *current* resolved home, rather
  than discarding it.
- Both `PluginManager._load_directory_module` (initial/`force=True`
  reload within the same home) and the shared `_clear_plugin_submodules`
  helper (profile switch / test teardown) evict `sys.modules[module_name]`
  **and every name prefixed with `module_name + "."`** before a plugin
  slug is (re-)imported, so relative-import submodules can never survive
  a reload or a home switch.
- Test isolation (`tests/conftest.py`'s `_hermetic_environment` fixture)
  calls a new `_reset_plugin_managers_for_tests()` helper that drops the
  entire keyed cache and purges every plugin submodule from `sys.modules`
  between tests, instead of only resetting the single-slot pointer.

Consequences:
- Per-profile LCM instances (and any other context-engine plugin) use
  their own `{home}/lcm.db` regardless of whether the profile switch went
  through `HERMES_HOME` or `set_hermes_home_override()`.
- Plugin discovery remains cached within a profile for normal
  performance, and re-entering a previously-seen profile reuses its
  cached manager instead of rebuilding from scratch.
- Sequential *and* interleaved profile switching — in tests, the gateway
  multiplexer worker, or embedded callers using the context-local
  override — no longer leaks context-engine state, plugin module state,
  or stale relative-import submodules across profiles.
- Regression coverage exercises the real production path
  (`set_hermes_home_override()`) rather than only the env-var path, and
  includes a dedicated relative-import leak test.
