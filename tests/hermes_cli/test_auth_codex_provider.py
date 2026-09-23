"""Tests for Codex auth — tokens stored in Hermes auth store (~/.hermes/auth.json)."""

import json
import time
import base64
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli.auth import (
    AuthError,
    DEFAULT_CODEX_BASE_URL,
    PROVIDER_REGISTRY,
    _read_codex_tokens,
    _save_codex_tokens,
    _import_codex_cli_tokens,
    _login_openai_codex,
    refresh_codex_oauth_pure,
    resolve_codex_runtime_credentials,
    resolve_provider,
)


def _setup_hermes_auth(hermes_home: Path, *, access_token: str = "access", refresh_token: str = "refresh"):
    """Write Codex tokens into the Hermes auth store."""
    hermes_home.mkdir(parents=True, exist_ok=True)
    auth_store = {
        "version": 1,
        "active_provider": "openai-codex",
        "providers": {
            "openai-codex": {
                "tokens": {
                    "access_token": access_token,
                    "refresh_token": refresh_token,
                },
                "last_refresh": "2026-02-26T00:00:00Z",
                "auth_mode": "chatgpt",
            },
        },
    }
    auth_file = hermes_home / "auth.json"
    auth_file.write_text(json.dumps(auth_store, indent=2))
    return auth_file


def _jwt_with_exp(exp_epoch: int) -> str:
    payload = {"exp": exp_epoch}
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode("utf-8")).rstrip(b"=").decode("utf-8")
    return f"h.{encoded}.s"






def test_resolve_codex_runtime_credentials_missing_access_token(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    _setup_hermes_auth(hermes_home, access_token="")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "missing-codex"))

    with pytest.raises(AuthError) as exc:
        resolve_codex_runtime_credentials()
    assert exc.value.code == "codex_auth_missing_access_token"
    assert exc.value.relogin_required is True


def test_resolve_codex_runtime_credentials_falls_back_to_pool_when_singleton_empty(tmp_path, monkeypatch):
    """Regression for #32992 — chat path returns 401 when singleton is empty but pool has creds.

    The chat path historically went through ``resolve_codex_runtime_credentials`` which
    only consulted ``providers.openai-codex.tokens`` and raised ``AuthError`` when that
    was empty.  The auxiliary path went through ``_read_codex_access_token`` which
    checks the pool first.  Users with creds only in the pool (manual seed, partial
    re-auth, restore from backup) hit a bare HTTP 401 on chat but worked fine on
    auxiliary calls.  The fallback closes that divergence.
    """
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    # Singleton: empty tokens (would normally raise AuthError).
    # Pool: valid access_token.
    auth_store = {
        "version": 1,
        "providers": {},  # no openai-codex singleton at all
        "credential_pool": {
            "openai-codex": [
                {
                    "source": "device_code",
                    "access_token": "pool-fallback-token",
                    "refresh_token": "pool-refresh",
                    "last_status": "ok",
                    "auth_type": "oauth",
                },
            ],
        },
    }
    (hermes_home / "auth.json").write_text(json.dumps(auth_store))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    resolved = resolve_codex_runtime_credentials()
    assert resolved["api_key"] == "pool-fallback-token"
    assert resolved["source"] == "credential_pool"
    assert resolved["base_url"]  # default codex backend URL


# ---------------------------------------------------------------------------
# Catalog account boundary: a named profile resolves the account its runtime
# actually runs on — the profile's own material first, else the global root's
# through ``_load_provider_state`` / ``read_credential_pool`` — and never
# another profile's, never a guess over an unreadable store.
# ---------------------------------------------------------------------------


def _codex_pool_row(access_token: str, *, source: str = "manual:device_code") -> dict:
    return {
        "id": f"id-{access_token}",
        "source": source,
        "auth_type": "oauth",
        "access_token": access_token,
        "refresh_token": f"{access_token}-refresh",
        "last_status": "ok",
    }


def _write_codex_store(home: Path, *, singleton=None, pool=None, raw=None) -> Path:
    home.mkdir(parents=True, exist_ok=True)
    auth_file = home / "auth.json"
    if raw is not None:
        auth_file.write_text(raw, encoding="utf-8")
        return auth_file
    store = {"version": 1, "providers": {}}
    if singleton is not None:
        store["providers"]["openai-codex"] = singleton
    if pool is not None:
        store["credential_pool"] = {"openai-codex": pool}
    auth_file.write_text(json.dumps(store), encoding="utf-8")
    return auth_file


def _singleton(access_token: str) -> dict:
    return {
        "tokens": {"access_token": access_token, "refresh_token": f"{access_token}-refresh"},
        "last_refresh": "2026-09-01T00:00:00Z",
        "auth_mode": "chatgpt",
    }


@pytest.fixture
def codex_profiles(tmp_path, monkeypatch):
    """A global root with two named profiles, scoped to ``worker``.

    ``HERMES_HOME`` is the root (so ``get_default_hermes_root()`` is the root)
    and the request scope is the context-local override, exactly as the
    dashboard's ``_profile_scope`` sets it.
    """
    import hermes_cli.auth as auth_mod
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    root = tmp_path / "root"
    worker = root / "profiles" / "worker"
    other = root / "profiles" / "other"
    for home in (root, worker, other):
        home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "no-codex-cli"))
    monkeypatch.setattr(auth_mod, "_global_auth_store_cache", None)
    token = set_hermes_home_override(str(worker))
    try:
        yield SimpleNamespace(root=root, worker=worker, other=other)
    finally:
        reset_hermes_home_override(token)


def _owning_root():
    from hermes_cli.web_server import _shared_codex_owning_root

    return _shared_codex_owning_root()


def _resolve_at(home: Path) -> dict:
    """The catalog account read exactly as ``_codex_catalog_at_owning_root``
    performs it: under the owning home's own scope."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    token = set_hermes_home_override(str(home))
    try:
        return resolve_codex_runtime_credentials()
    finally:
        reset_hermes_home_override(token)


def _catalog_of(access_token) -> list:
    """The model ids the fake account API lists for *access_token*."""
    return [f"model-of-{access_token}"]


def _record_codex_catalog_fetches(monkeypatch) -> list:
    """Patch the Codex account-catalog fetcher to report which token it was asked for.

    Both catalog paths end in ``get_codex_model_ids``: the owning home's own
    read (via ``hermes_cli.models._codex_catalog``, with that home's resolved
    token) and the selected-token fallback. Returns the list of tokens asked.
    """
    asked = []

    def fake_get_codex_model_ids(access_token=None):
        asked.append(access_token)
        return _catalog_of(access_token)

    monkeypatch.setattr(
        "hermes_cli.codex_models.get_codex_model_ids", fake_get_codex_model_ids
    )
    return asked


def _runtime_selected_token():
    """What the scoped profile's runtime selects, via the production selector's
    read-only replica (``load_pool`` + ``select``), and whether it is borrowed."""
    from agent.credential_pool import preview_runtime_selection

    entry, borrowed = preview_runtime_selection("openai-codex")
    return (entry.access_token if entry is not None else None), borrowed


def _route_codex_catalog(*, refresh: bool = False):
    """The Codex catalog the owner's Models route lists for the scoped profile,
    composed exactly as ``get_model_options`` composes it."""
    from hermes_cli.web_server import (
        _codex_catalog_for_runtime_account,
        _codex_runtime_account,
    )

    account = _codex_runtime_account()
    if account is None:
        return None, None
    home, token = account
    return account, _codex_catalog_for_runtime_account(home, token, refresh=refresh)


def test_profile_without_codex_material_is_owned_by_the_roots_account(codex_profiles):
    """The operator's defect: the runtime ran on the root's pool grant.

    ``read_credential_pool`` hands a profile with zero Codex entries the root's
    slice, and ``_load_provider_state_with_source`` finds no singleton of the
    profile's own, so the root owns the account the profile runs on and the
    owner's catalog read is taken there — resolving the root's pool account
    rather than raising ``codex_auth_missing`` (which degraded the picker to
    the static list and dropped account-gated models such as Astra).

    An ORDINARY profile-scoped read stays isolated and does not borrow the
    root's pool (the approved behaviour pinned by
    ``test_generic_profile_reads_keep_their_own_isolated_catalog``).
    """
    from hermes_cli.auth import read_credential_pool

    _write_codex_store(codex_profiles.root, pool=[_codex_pool_row("root-pool-at")])

    assert [row["access_token"] for row in read_credential_pool("openai-codex")] == [
        "root-pool-at"
    ]
    assert _owning_root() == codex_profiles.root
    resolved = _resolve_at(_owning_root())
    assert (resolved["api_key"], resolved["source"]) == ("root-pool-at", "credential_pool")
    with pytest.raises(AuthError) as exc:
        resolve_codex_runtime_credentials()
    assert exc.value.code == "codex_auth_missing"
    # Reading never copies the root's material into the profile.
    assert not (codex_profiles.worker / "auth.json").exists()


def test_malformed_profile_store_keeps_its_corrupt_copy_and_reads_as_empty(codex_profiles):
    """(a) Malformed JSON is preserved on disk, never discarded or rewritten.

    ``_load_auth_store`` copies the bytes to ``auth.json.corrupt`` and reads the
    store as empty — the runtime's existing behaviour — so the account is
    decided exactly as the runtime decides it (here: the root's pool, which a
    profile with no readable entries of its own borrows).
    """
    garbage = '{"providers": {"openai-codex": {"tokens": '
    auth_file = _write_codex_store(codex_profiles.worker, raw=garbage)
    _write_codex_store(codex_profiles.root, pool=[_codex_pool_row("root-pool-at")])

    assert _owning_root() == codex_profiles.root
    assert _resolve_at(codex_profiles.root)["api_key"] == "root-pool-at"
    # The profile's own read neither guesses the lost tokens nor borrows.
    with pytest.raises(AuthError):
        resolve_codex_runtime_credentials()

    corrupt = codex_profiles.worker / "auth.json.corrupt"
    assert corrupt.read_text(encoding="utf-8") == garbage
    # The original is still on disk, byte for byte: nothing was discarded.
    assert auth_file.read_text(encoding="utf-8") == garbage


def test_unreadable_profile_store_is_refused_rather_than_guessed(codex_profiles):
    """(b) A store that exists but cannot be read is not an empty store.

    Mechanism: ``auth.json`` is a DIRECTORY, so ``read_text`` raises
    ``IsADirectoryError`` (an ``OSError``). ``chmod 000`` is deliberately NOT
    used: this suite runs as root, which reads a mode-000 file anyway, so a
    chmod-based test would silently prove nothing.
    """
    (codex_profiles.worker / "auth.json").mkdir()
    root_file = _write_codex_store(
        codex_profiles.root,
        singleton=_singleton("root-at"),
        pool=[_codex_pool_row("root-pool-at")],
    )
    before = root_file.read_bytes()

    # Refused: neither degraded to "no credentials" nor silently swapped for
    # the root's grant.
    with pytest.raises(OSError):
        resolve_codex_runtime_credentials()
    assert _owning_root() is None
    assert root_file.read_bytes() == before
    assert (codex_profiles.worker / "auth.json").is_dir()


def test_own_singleton_plus_borrowed_root_pool_resolves_the_profiles_own_grant(
    codex_profiles, monkeypatch,
):
    """(c) Catalog account == the account the runtime actually selects; no store mutates.

    The profile's own singleton-first read resolves its own grant
    (``worker-at``), but the runtime runs on ``load_pool().select()``, which
    puts the borrowed root pool row (priority 0) ahead of the profile's seeded
    singleton and selects ``root-pool-at``. The catalog must therefore be
    ``root-pool-at``'s: not the profile singleton's (the account's own read),
    and not the root singleton's (what the root's own read would resolve).
    Only model ids cross the boundary: neither auth store changes by a byte,
    and none of the root's material reaches the profile.
    """
    from hermes_cli.auth import read_credential_pool

    worker_file = _write_codex_store(
        codex_profiles.worker, singleton=_singleton("worker-at")
    )
    root_file = _write_codex_store(
        codex_profiles.root,
        singleton=_singleton("root-at"),
        pool=[_codex_pool_row("root-pool-at")],
    )
    worker_before, root_before = worker_file.read_bytes(), root_file.read_bytes()
    asked = _record_codex_catalog_fetches(monkeypatch)

    assert [row["access_token"] for row in read_credential_pool("openai-codex")] == [
        "root-pool-at"
    ]
    # The profile's own read is unchanged: its own singleton, singleton first.
    resolved = resolve_codex_runtime_credentials()
    assert (resolved["api_key"], resolved["source"]) == ("worker-at", "hermes-auth-store")

    # The runtime selects the borrowed root pool row, not the profile's grant.
    assert _runtime_selected_token() == ("root-pool-at", True)
    assert _owning_root() == codex_profiles.root
    # The root's own read would list a different grant (its singleton) ...
    assert _resolve_at(codex_profiles.root)["api_key"] == "root-at"

    # ... so the route asks the catalog for exactly the selected account.
    account, models = _route_codex_catalog()
    assert account == (codex_profiles.root, "root-pool-at")
    assert models == _catalog_of("root-pool-at")
    assert asked == ["root-pool-at"]

    assert worker_file.read_bytes() == worker_before
    assert root_file.read_bytes() == root_before
    assert not (codex_profiles.worker / "provider_models_cache.json").exists()


def test_profile_with_only_its_own_pool_never_borrows_the_roots_pool(codex_profiles):
    """(d) Any pool entry of the profile's own shadows the root's pool slice."""
    _write_codex_store(
        codex_profiles.worker, pool=[_codex_pool_row("worker-pool-at")],
    )
    _write_codex_store(codex_profiles.root, pool=[_codex_pool_row("root-pool-at")])

    resolved = resolve_codex_runtime_credentials()
    assert (resolved["api_key"], resolved["source"]) == (
        "worker-pool-at", "credential_pool",
    )
    assert _owning_root() is None


def test_profile_and_root_with_no_account_claim_nothing(codex_profiles):
    """(e) No account anywhere: the catalog read raises and no owner is claimed."""
    with pytest.raises(AuthError) as exc:
        resolve_codex_runtime_credentials()
    assert exc.value.code == "codex_auth_missing"
    assert _owning_root() is None
    assert not (codex_profiles.worker / "auth.json").exists()


def test_a_foreign_profiles_grant_is_never_borrowed(codex_profiles):
    """(f) The only fallback is the global root — never a sibling profile.

    ``other`` holds a full Codex account; ``worker`` holds nothing. Nothing of
    ``other``'s may reach ``worker``: with an empty root there is no account at
    all, and once the root has one, the root's account is what resolves.
    """
    _write_codex_store(
        codex_profiles.other,
        singleton=_singleton("other-at"),
        pool=[_codex_pool_row("other-pool-at")],
    )

    with pytest.raises(AuthError) as exc:
        resolve_codex_runtime_credentials()
    assert exc.value.code == "codex_auth_missing"
    assert _owning_root() is None

    _write_codex_store(codex_profiles.root, pool=[_codex_pool_row("root-pool-at")])
    owner = _owning_root()
    assert owner == codex_profiles.root
    assert owner != codex_profiles.other
    assert _resolve_at(owner)["api_key"] == "root-pool-at"
    # And the scoped profile's own read still reaches nobody else's account.
    with pytest.raises(AuthError) as exc:
        resolve_codex_runtime_credentials()
    assert exc.value.code == "codex_auth_missing"


def test_a_singleton_grant_alone_never_establishes_the_account(
    codex_profiles, monkeypatch,
):
    """The root holding a singleton is not, by itself, the profile's account.

    The profile carries its own (empty) ``providers.openai-codex`` block, which
    ``_load_provider_state_with_source`` resolves profile-first: it SHADOWS the
    root's singleton for this profile. The runtime reaches the root only
    through the borrowed POOL and selects ``root-pool-at``. Invariant: the
    catalog account equals that runtime-selected account — never the root's
    SINGLETON, which is what re-reading the catalog at the root would resolve —
    and neither auth store is mutated to get there.
    """
    worker_file = _write_codex_store(codex_profiles.worker, singleton={})
    root_file = _write_codex_store(
        codex_profiles.root,
        singleton=_singleton("root-singleton-at"),
        pool=[_codex_pool_row("root-pool-at")],
    )
    worker_before, root_before = worker_file.read_bytes(), root_file.read_bytes()
    asked = _record_codex_catalog_fetches(monkeypatch)

    # What the root's own read would have picked: its singleton, not the pool.
    assert _resolve_at(codex_profiles.root)["api_key"] == "root-singleton-at"
    # The profile's own read still resolves nothing: its empty block shadows
    # the root's singleton and its pool fallback stays profile-local.
    with pytest.raises(AuthError):
        resolve_codex_runtime_credentials()

    # The runtime runs on the borrowed pool row, so that is the catalog account.
    assert _runtime_selected_token() == ("root-pool-at", True)
    account, models = _route_codex_catalog()
    assert account == (codex_profiles.root, "root-pool-at")
    assert models == _catalog_of("root-pool-at")
    assert asked == ["root-pool-at"]
    assert "root-singleton-at" not in asked

    assert worker_file.read_bytes() == worker_before
    assert root_file.read_bytes() == root_before
    assert not (codex_profiles.worker / "provider_models_cache.json").exists()


def test_catalog_is_the_runtime_selected_borrowed_pool_account_not_either_singleton(
    codex_profiles, monkeypatch,
):
    """Three-way mix: root pool A (priority 0, borrowed), root singleton B, profile singleton C.

    The runtime (``load_pool().select()``) runs on A. The profile's own read
    resolves C and the root's own read resolves B, so a route that listed
    either read's catalog would show an account the runtime does not run on.
    Invariant: the catalog returned is A's — not B's, not C's, and not the
    foreign profile's — and no auth store changes by a byte.
    """
    worker_file = _write_codex_store(
        codex_profiles.worker, singleton=_singleton("profile-singleton-c-at")
    )
    root_file = _write_codex_store(
        codex_profiles.root,
        singleton=_singleton("root-singleton-b-at"),
        pool=[dict(_codex_pool_row("root-pool-a-at"), priority=0)],
    )
    other_file = _write_codex_store(
        codex_profiles.other,
        singleton=_singleton("other-at"),
        pool=[_codex_pool_row("other-pool-at")],
    )
    before = {f: f.read_bytes() for f in (worker_file, root_file, other_file)}
    asked = _record_codex_catalog_fetches(monkeypatch)

    # The two reads a regressed route could list instead.
    assert resolve_codex_runtime_credentials()["api_key"] == "profile-singleton-c-at"
    assert _resolve_at(codex_profiles.root)["api_key"] == "root-singleton-b-at"

    assert _runtime_selected_token() == ("root-pool-a-at", True)
    account, models = _route_codex_catalog()
    assert account == (codex_profiles.root, "root-pool-a-at")
    assert models == _catalog_of("root-pool-a-at")
    assert models != _catalog_of("root-singleton-b-at")
    assert models != _catalog_of("profile-singleton-c-at")
    assert asked == ["root-pool-a-at"]

    for auth_file, raw in before.items():
        assert auth_file.read_bytes() == raw
    assert not (codex_profiles.worker / "provider_models_cache.json").exists()


def test_a_borrowed_root_singleton_is_refreshed_in_the_roots_store(
    codex_profiles, monkeypatch,
):
    """Refreshing the root's single-use grant from a profile must not fork it.

    The rotated chain lands back in the root's auth.json (where the grant was
    read) and the profile never acquires a copy of the root's tokens.
    """
    expiring = _jwt_with_exp(int(time.time()) - 60)
    fresh = _jwt_with_exp(int(time.time()) + 3600)
    root_file = _write_codex_store(codex_profiles.root, singleton=_singleton(expiring))
    calls = []

    def fake_refresh(access_token, refresh_token, *, timeout_seconds=20.0):
        calls.append(refresh_token)
        return {
            "access_token": fresh,
            "refresh_token": "rotated-refresh",
            "last_refresh": "2026-09-23T00:00:00Z",
        }

    monkeypatch.setattr("hermes_cli.auth.refresh_codex_oauth_pure", fake_refresh)

    resolved = resolve_codex_runtime_credentials()

    assert resolved["api_key"] == fresh
    assert calls == [f"{expiring}-refresh"]
    root_tokens = json.loads(root_file.read_text())["providers"]["openai-codex"]["tokens"]
    assert root_tokens == {"access_token": fresh, "refresh_token": "rotated-refresh"}
    worker_file = codex_profiles.worker / "auth.json"
    if worker_file.exists():
        assert "openai-codex" not in json.loads(worker_file.read_text()).get("providers", {})


def test_catalog_at_owning_root_reads_and_refreshes_only_the_roots_cache(
    codex_profiles, monkeypatch,
):
    """Only model ids cross the scope boundary; fetch, cache and SWR stay at root.

    A stale root cache row is served and its background refresh runs the
    fetcher under the ROOT's home and writes the ROOT's cache — not the
    profile's, and not whatever home the refresh thread would otherwise
    inherit.
    """
    import hermes_cli.models as models_mod
    from hermes_cli.web_server import _codex_catalog_at_owning_root
    from hermes_constants import get_hermes_home

    _write_codex_store(codex_profiles.root, pool=[_codex_pool_row("root-pool-at")])
    fetched_under = []

    def fake_live(provider, force_refresh=False):
        fetched_under.append((provider, str(get_hermes_home())))
        return ["gpt-6-sol", "gpt-6-astra"]

    monkeypatch.setattr(models_mod, "provider_model_ids", fake_live)
    root = _owning_root()
    assert root == codex_profiles.root

    # Cold: a blocking fetch under the root's home, persisted in its cache.
    assert _codex_catalog_at_owning_root(root, refresh=False) == [
        "gpt-6-sol", "gpt-6-astra",
    ]
    assert fetched_under == [("openai-codex", str(codex_profiles.root))]
    root_cache = codex_profiles.root / "provider_models_cache.json"
    rows = json.loads(root_cache.read_text())
    assert rows["openai-codex"]["models"] == ["gpt-6-sol", "gpt-6-astra"]

    # Stale: served immediately, refreshed off-thread — still at the root.
    rows["openai-codex"]["at"] = time.time() - 7200
    rows["openai-codex"]["models"] = ["gpt-6-sol"]
    root_cache.write_text(json.dumps(rows))
    fetched_under.clear()
    assert _codex_catalog_at_owning_root(root, refresh=False) == ["gpt-6-sol"]
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        with models_mod._swr_refresh_lock:
            if not models_mod._swr_refresh_inflight:
                break
        time.sleep(0.01)
    assert fetched_under == [("openai-codex", str(codex_profiles.root))]
    assert json.loads(root_cache.read_text())["openai-codex"]["models"] == [
        "gpt-6-sol", "gpt-6-astra",
    ]

    # The profile got neither a cache row nor any token material.
    assert not (codex_profiles.worker / "provider_models_cache.json").exists()
    assert not (codex_profiles.worker / "auth.json").exists()




def test_save_codex_tokens_syncs_credential_pool(tmp_path, monkeypatch):
    """Re-auth must update the credential_pool device_code entry, not just providers.

    Regression for #33000: the runtime selects from credential_pool, so a
    re-auth that only refreshed providers.openai-codex.tokens left the pool
    holding a consumed refresh token and stale error markers, causing an
    immediate 401 token_invalidated on the next request.
    """
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(json.dumps({
        "version": 1,
        "providers": {
            "openai-codex": {
                "tokens": {"access_token": "old-at", "refresh_token": "old-rt"},
                "last_refresh": "2026-01-01T00:00:00Z",
                "auth_mode": "chatgpt",
            },
        },
        "credential_pool": {
            "openai-codex": [
                {
                    "id": "abc123",
                    "source": "device_code",
                    "auth_type": "oauth",
                    "access_token": "old-at",
                    "refresh_token": "old-rt",
                    "last_status": "exhausted",
                    "last_error_code": 401,
                    "last_error_reason": "token_invalidated",
                    "last_error_reset_at": 9999999999,
                },
                {
                    "id": "manual1",
                    "source": "manual:codex",
                    "auth_type": "oauth",
                    "access_token": "manual-at",
                    "refresh_token": "manual-rt",
                },
            ],
        },
    }))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    _save_codex_tokens({"access_token": "new-at", "refresh_token": "new-rt"},
                       last_refresh="2026-05-27T00:00:00Z")

    auth = json.loads((hermes_home / "auth.json").read_text())
    pool = auth["credential_pool"]["openai-codex"]
    seeded = next(e for e in pool if e["source"] == "device_code")
    assert seeded["access_token"] == "new-at"
    assert seeded["refresh_token"] == "new-rt"
    assert seeded["last_refresh"] == "2026-05-27T00:00:00Z"
    assert seeded["last_status"] is None
    assert seeded["last_error_code"] is None
    assert seeded["last_error_reason"] is None
    assert seeded["last_error_reset_at"] is None

    # Manual entries are independent credentials and must not be overwritten.
    manual = next(e for e in pool if e["source"] == "manual:codex")
    assert manual["access_token"] == "manual-at"
    assert manual["refresh_token"] == "manual-rt"

    # Provider singleton is updated too.
    assert auth["providers"]["openai-codex"]["tokens"]["access_token"] == "new-at"


def test_save_codex_tokens_syncs_manual_device_code_entries(tmp_path, monkeypatch):
    """Re-auth must refresh ``manual:device_code`` entries that are true
    aliases of the singleton, while leaving INDEPENDENT entries alone.

    Original regression for #33538: a user who hit #33000 before the #33164
    fix landed would have run ``hermes auth add openai-codex`` as a
    workaround, leaving a pool entry with ``source="manual:device_code"``.
    On every subsequent re-auth via setup/model picker, the singleton-seeded
    ``device_code`` entry got refreshed but the ``manual:device_code`` entry
    stayed stale, recreating the same 401 token_invalidated symptom that
    #33164 was supposed to fix.

    Narrowed for #39236: the original fix treated every ``manual:device_code``
    entry as a singleton-alias and refreshed them all, which silently
    clobbered independent accounts added via ``hermes auth add openai-codex``.
    The current behavior refreshes only entries whose access_token matches
    the *previous* singleton access_token (true legacy aliases), and leaves
    distinct-token entries alone (independent accounts).
    """
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(json.dumps({
        "version": 1,
        "providers": {
            "openai-codex": {
                "tokens": {"access_token": "old-at", "refresh_token": "old-rt"},
                "last_refresh": "2026-01-01T00:00:00Z",
                "auth_mode": "chatgpt",
            },
        },
        "credential_pool": {
            "openai-codex": [
                {
                    "id": "seeded",
                    "source": "device_code",
                    "auth_type": "oauth",
                    "access_token": "old-at",
                    "refresh_token": "old-rt",
                },
                # Legacy alias from the #33000 workaround era — its tokens
                # match the singleton, so it is a true alias and SHOULD be
                # refreshed (preserves #33538 behavior).
                {
                    "id": "legacy-alias",
                    "source": "manual:device_code",
                    "auth_type": "oauth",
                    "access_token": "old-at",
                    "refresh_token": "old-rt",
                    "last_status": "exhausted",
                    "last_error_code": 401,
                    "last_error_reason": "token_invalidated",
                },
                # Independent account from `hermes auth add openai-codex` —
                # its tokens are distinct from the singleton.  Must NOT be
                # overwritten by a re-auth that targeted a different account
                # (#39236).
                {
                    "id": "independent",
                    "source": "manual:device_code",
                    "auth_type": "oauth",
                    "access_token": "independent-at",
                    "refresh_token": "independent-rt",
                },
                {
                    "id": "api-key",
                    "source": "manual:api_key",
                    "auth_type": "api_key",
                    "access_token": "user-api-key",
                },
            ],
        },
    }))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    _save_codex_tokens({"access_token": "fresh-at", "refresh_token": "fresh-rt"},
                       last_refresh="2026-05-28T00:00:00Z")

    auth = json.loads((hermes_home / "auth.json").read_text())
    pool = auth["credential_pool"]["openai-codex"]

    # Singleton-seeded device_code entry: refreshed and error markers cleared.
    seeded = next(e for e in pool if e["id"] == "seeded")
    assert seeded["access_token"] == "fresh-at"
    assert seeded["refresh_token"] == "fresh-rt"

    # Legacy alias (tokens matched previous singleton): ALSO refreshed.
    legacy = next(e for e in pool if e["id"] == "legacy-alias")
    assert legacy["access_token"] == "fresh-at"
    assert legacy["refresh_token"] == "fresh-rt"
    assert legacy["last_refresh"] == "2026-05-28T00:00:00Z"
    assert legacy["last_status"] is None
    assert legacy["last_error_code"] is None
    assert legacy["last_error_reason"] is None

    # Independent manual:device_code entry: NOT overwritten (#39236).
    independent = next(e for e in pool if e["id"] == "independent")
    assert independent["access_token"] == "independent-at"
    assert independent["refresh_token"] == "independent-rt"

    # manual:api_key entry: untouched — independent credential.
    api_key = next(e for e in pool if e["source"] == "manual:api_key")
    assert api_key["access_token"] == "user-api-key"
    assert "refresh_token" not in api_key or api_key.get("refresh_token") is None


def test_save_codex_tokens_does_not_overwrite_independent_manual_entries(tmp_path, monkeypatch):
    """Re-auth must NOT overwrite ``manual:device_code`` entries that hold
    independent token material (different OpenAI/ChatGPT accounts).

    Regression for #39236: ``hermes auth add openai-codex`` for accounts B and C
    routes through ``_save_codex_tokens`` because the singleton path is the
    only Codex OAuth save flow.  The #33538 fix refreshed every
    ``manual:device_code`` entry on every re-auth, which works fine for the
    one-account/legacy-workaround case but silently overwrote distinct
    independent accounts with the latest-authenticated tokens (labels
    preserved, token material clobbered, status/quota readings then lie).

    The safe invariant: an entry is a singleton-alias only when its current
    access_token matches the *previous* singleton access_token.  Manual
    entries whose tokens never matched the singleton are independent accounts
    and must be left alone.
    """
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(json.dumps({
        "version": 1,
        "providers": {
            "openai-codex": {
                # Old singleton tokens — represent "account A" which the user
                # logged in with via setup originally.
                "tokens": {"access_token": "acctA-at", "refresh_token": "acctA-rt"},
                "last_refresh": "2026-01-01T00:00:00Z",
                "auth_mode": "chatgpt",
                "label": "account-A",
            },
        },
        "credential_pool": {
            "openai-codex": [
                # The seeded singleton mirror of account A.
                {
                    "id": "seeded",
                    "label": "account-A",
                    "source": "device_code",
                    "auth_type": "oauth",
                    "access_token": "acctA-at",
                    "refresh_token": "acctA-rt",
                },
                # Two INDEPENDENT manual entries added later via
                # ``hermes auth add openai-codex`` (account B and account C).
                # Each has its OWN distinct token material, unrelated to the
                # singleton.
                {
                    "id": "acctB",
                    "label": "account-B",
                    "source": "manual:device_code",
                    "auth_type": "oauth",
                    "access_token": "acctB-at",
                    "refresh_token": "acctB-rt",
                },
                {
                    "id": "acctC",
                    "label": "account-C",
                    "source": "manual:device_code",
                    "auth_type": "oauth",
                    "access_token": "acctC-at",
                    "refresh_token": "acctC-rt",
                },
            ],
        },
    }))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    # User re-authenticates account A — fresh device-code login produces new
    # tokens.  The legitimate update is the seeded singleton mirror; the
    # independent acctB/acctC entries must be untouched.
    _save_codex_tokens(
        {"access_token": "acctA-new-at", "refresh_token": "acctA-new-rt"},
        last_refresh="2026-06-05T00:00:00Z",
    )

    auth = json.loads((hermes_home / "auth.json").read_text())
    pool = auth["credential_pool"]["openai-codex"]

    # Singleton-seeded entry: refreshed (legitimate sync).
    seeded = next(e for e in pool if e["source"] == "device_code")
    assert seeded["access_token"] == "acctA-new-at"
    assert seeded["refresh_token"] == "acctA-new-rt"
    assert seeded["last_refresh"] == "2026-06-05T00:00:00Z"

    # acctB: INDEPENDENT entry — must NOT be overwritten.
    acctB = next(e for e in pool if e["id"] == "acctB")
    assert acctB["access_token"] == "acctB-at", (
        "acctB was clobbered by acctA re-auth (#39236 regression)"
    )
    assert acctB["refresh_token"] == "acctB-rt"

    # acctC: INDEPENDENT entry — must NOT be overwritten.
    acctC = next(e for e in pool if e["id"] == "acctC")
    assert acctC["access_token"] == "acctC-at", (
        "acctC was clobbered by acctA re-auth (#39236 regression)"
    )
    assert acctC["refresh_token"] == "acctC-rt"


def test_save_codex_tokens_clears_error_markers_only_on_refreshed_entries(tmp_path, monkeypatch):
    """Error markers must be cleared only on entries that were actually
    refreshed by this re-auth.  Independent ``manual:device_code`` entries
    with their own stale-error markers must be left alone (their stale state
    is not the current re-auth's business).
    """
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(json.dumps({
        "version": 1,
        "providers": {
            "openai-codex": {
                "tokens": {"access_token": "acctA-at", "refresh_token": "acctA-rt"},
                "auth_mode": "chatgpt",
            },
        },
        "credential_pool": {
            "openai-codex": [
                {
                    "id": "seeded",
                    "source": "device_code",
                    "auth_type": "oauth",
                    "access_token": "acctA-at",
                    "refresh_token": "acctA-rt",
                    "last_status": "exhausted",
                    "last_error_code": 401,
                },
                {
                    "id": "acctB",
                    "source": "manual:device_code",
                    "auth_type": "oauth",
                    "access_token": "acctB-at",
                    "refresh_token": "acctB-rt",
                    "last_status": "exhausted",
                    "last_error_code": 429,
                    "last_error_reason": "quota_exhausted",
                },
            ],
        },
    }))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    _save_codex_tokens(
        {"access_token": "fresh-at", "refresh_token": "fresh-rt"},
        last_refresh="2026-06-05T00:00:00Z",
    )

    auth = json.loads((hermes_home / "auth.json").read_text())
    pool = auth["credential_pool"]["openai-codex"]

    # Singleton: refreshed AND error markers cleared.
    seeded = next(e for e in pool if e["id"] == "seeded")
    assert seeded["access_token"] == "fresh-at"
    assert seeded["last_status"] is None
    assert seeded["last_error_code"] is None

    # Independent acctB: NOT refreshed AND error markers NOT cleared.
    # (Its 429 quota state belongs to acctB's own account, not acctA's re-auth.)
    acctB = next(e for e in pool if e["id"] == "acctB")
    assert acctB["access_token"] == "acctB-at"  # not overwritten
    assert acctB["last_status"] == "exhausted"  # not cleared
    assert acctB["last_error_code"] == 429
    assert acctB["last_error_reason"] == "quota_exhausted"




def test_codex_tokens_not_written_to_shared_file(tmp_path, monkeypatch):
    """Verify _save_codex_tokens writes only to Hermes auth store, not ~/.codex/."""
    hermes_home = tmp_path / "hermes"
    codex_home = tmp_path / "codex-cli"
    hermes_home.mkdir(parents=True, exist_ok=True)
    codex_home.mkdir(parents=True, exist_ok=True)

    (hermes_home / "auth.json").write_text(json.dumps({"version": 1, "providers": {}}))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    _save_codex_tokens({"access_token": "hermes-at", "refresh_token": "hermes-rt"})

    # ~/.codex/auth.json should NOT exist — _save_codex_tokens only touches Hermes store
    assert not (codex_home / "auth.json").exists()

    # Hermes auth store should have the tokens
    data = _read_codex_tokens()
    assert data["tokens"]["access_token"] == "hermes-at"


def test_resolve_returns_hermes_auth_store_source(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    _setup_hermes_auth(hermes_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    creds = resolve_codex_runtime_credentials()
    assert creds["source"] == "hermes-auth-store"
    assert creds["provider"] == "openai-codex"
    assert creds["base_url"] == DEFAULT_CODEX_BASE_URL


class _StubHTTPResponse:
    def __init__(self, status_code: int, payload, headers=None):
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}
        self.text = json.dumps(payload) if isinstance(payload, (dict, list)) else str(payload)

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class _StubHTTPClient:
    def __init__(self, response):
        self._response = response

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def post(self, *args, **kwargs):
        return self._response


def _patch_httpx(monkeypatch, response):
    def _factory(*args, **kwargs):
        return _StubHTTPClient(response)

    monkeypatch.setattr("hermes_cli.auth.httpx.Client", _factory)




def test_refresh_429_classified_as_quota_not_auth_failure(monkeypatch):
    """429 from the token endpoint is a usage-quota cap, not an auth failure.

    Regression test for #32790: must NOT force relogin and must carry the
    dedicated rate-limit code so callers surface a "retry later" notice rather
    than a misleading "run hermes auth".
    """
    from hermes_cli.auth import (
        CODEX_RATE_LIMITED_CODE,
        format_auth_error,
        is_rate_limited_auth_error,
    )

    response = _StubHTTPResponse(
        429,
        {"error": {"message": "You hit your usage limit.", "code": "usage_limit_reached"}},
        headers={"retry-after": "120"},
    )
    _patch_httpx(monkeypatch, response)

    with pytest.raises(AuthError) as exc_info:
        refresh_codex_oauth_pure("a-tok", "r-tok")

    err = exc_info.value
    assert err.code == CODEX_RATE_LIMITED_CODE
    assert err.relogin_required is False
    assert is_rate_limited_auth_error(err) is True
    assert "retry after 120s" in str(err)
    # User-facing copy must not tell the operator to re-authenticate.
    rendered = format_auth_error(err)
    assert "re-authenticate" not in rendered
    assert "hermes auth" not in rendered


def test_refresh_429_without_retry_after_header(monkeypatch):
    """429 without a Retry-After header still classifies as quota, no relogin."""
    from hermes_cli.auth import CODEX_RATE_LIMITED_CODE

    response = _StubHTTPResponse(429, {"error": "rate_limited"})
    _patch_httpx(monkeypatch, response)

    with pytest.raises(AuthError) as exc_info:
        refresh_codex_oauth_pure("a-tok", "r-tok")

    err = exc_info.value
    assert err.code == CODEX_RATE_LIMITED_CODE
    assert err.relogin_required is False
    assert "quota exhausted" in str(err).lower()


def test_is_rate_limited_auth_error_distinguishes_credential_errors():
    """Missing/expired credentials must NOT be treated as rate-limit errors."""
    from hermes_cli.auth import CODEX_RATE_LIMITED_CODE, is_rate_limited_auth_error

    rate_limited = AuthError(
        "quota", provider="openai-codex", code=CODEX_RATE_LIMITED_CODE, relogin_required=False
    )
    missing_creds = AuthError(
        "No Codex credentials stored.",
        provider="openai-codex",
        code="codex_auth_missing",
        relogin_required=True,
    )
    assert is_rate_limited_auth_error(rate_limited) is True
    assert is_rate_limited_auth_error(missing_creds) is False
    assert is_rate_limited_auth_error(ValueError("nope")) is False




class _FakeResp:
    def __init__(self, status_code, json_data=None, headers=None):
        self.status_code = status_code
        self._json = json_data or {}
        self.headers = headers or {}

    def json(self):
        return self._json


def _patch_httpx_post(monkeypatch, responses):
    """Patch hermes_cli.auth.httpx.Client so .post() returns queued responses."""
    seq = iter(responses)

    class _FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, *args, **kwargs):
            return next(seq)

    monkeypatch.setattr("hermes_cli.auth.httpx.Client", lambda *a, **k: _FakeClient())




