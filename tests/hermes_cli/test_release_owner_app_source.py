"""The owner app's merge commit fetched on the release host, and its archive (owner app release plan, card 2).

A local bare repository laid out like GitHub stands in for the owner app repository, with the transport's
GIT_ORIGIN pointed at it over file://, and the transport tests' stand-in plays the token endpoint, recording each
token request. The App key is a throwaway made in the test and placed where the transport reads it, under the
platform root, and the only token is the stand-in's fake one.
"""
import importlib
import io
import stat
import subprocess
import tarfile
import tempfile
from pathlib import Path

import pytest

from tests.hermes_cli.test_kanban_delivery_github import (  # noqa: F401  (github is a fixture)
    _RUN, BASIC, DUMMY, SAMPLE, TOKEN, _block_sockets, _carries, _files_text, _git, _module, _new_key, _pem,
    _record_git, _record_mkdtemp, _subcommand, _transport_module, github,
)

OWNER_APP = "mtbitcr/raphael-workspace"
PLATFORM = "mtbitcr/hermes-agent"  # the platform repository, of which the step reads nothing
CONTENTS_READ = {"contents": "read"}
OWNER_APP_TOKEN = {"repositories": ["raphael-workspace"], "permissions": CONTENTS_READ}
KEY = Path("secrets") / "github-app" / "raphael-agent-factory.pem"  # under the platform root, where the App key is


def _source_module():
    """Import inside each test, so a missing module fails that test rather than the whole file."""
    return importlib.import_module("hermes_cli.release_owner_app_source")


def _expected_copy(root):
    """Where the owner app copy belongs, worked out here from the root rather than asked of the module."""
    return root / "kanban" / "release-source" / "raphael-workspace.git"


def _owner_app(tmp_path, card=None):
    """The owner app repository on a bare remote laid out like GitHub: a real merge of a card branch with a tag on it,
    main one commit past it, and a branch beside both, so a fetch of anything more than the merge would show. The card
    commit also holds the files of `card`, by name. Returns the origin, the remote and the commits by name."""
    work = tmp_path / "owner-app"
    _git("init", "-q", "-b", "main", str(work))
    identity = ("-c", "user.name=t", "-c", "user.email=t@example.invalid", "-c", "commit.gpgsign=false")

    def commit(message, files):
        for name, text in files.items():
            (work / name).parent.mkdir(parents=True, exist_ok=True)
            (work / name).write_text(text)
        _git("add", "-A", cwd=work)
        _git(*identity, "commit", "-q", "-m", message, cwd=work)
        return _git("rev-parse", "HEAD", cwd=work)

    (work / "scripts").mkdir()
    (work / "scripts" / "start.sh").write_text("#!/bin/sh\nexit 0\n")
    (work / "scripts" / "start.sh").chmod(0o755)  # a mode the archive must keep
    (work / "app").mkdir()
    (work / "app" / "current.html").symlink_to("page.html")  # and a link
    commits = {"base": commit("base", {"README.md": "# raphael-workspace\n", "app/page.html": "<p>one</p>\n"})}
    _git("checkout", "-q", "-b", "card", cwd=work)
    commits["card"] = commit("card", {"app/page.html": "<p>two</p>\n", "app/deep/nested/data.json": "{}\n",
                                      **(card or {})})
    _git("checkout", "-q", "main", cwd=work)
    commits["beside"] = commit("beside", {"docs/notes.md": "notes\n"})
    _git(*identity, "merge", "-q", "--no-ff", "-m", "Merge card", "card", cwd=work)
    commits["merge"] = _git("rev-parse", "HEAD", cwd=work)
    _git(*identity, "tag", "-a", "-m", "release", "v1", cwd=work)  # an annotated tag on the merge itself
    commits["later"] = commit("later", {"later.txt": "after the merge\n"})
    _git("checkout", "-q", "-b", "other", commits["base"], cwd=work)
    commits["other"] = commit("other", {"other.txt": "beside main\n"})
    remotes = tmp_path / "remotes"
    bare = remotes / "mtbitcr" / "raphael-workspace.git"
    _git("init", "-q", "--bare", str(bare))
    _git("push", "-q", str(bare), "main", "card", "other", "v1", cwd=work)
    return remotes.as_uri(), bare, commits


def _release_host(github, monkeypatch, tmp_path, card=None):
    """The transport pointed at the stand-in and the owner app's origin, the platform root at tmp_path/root with the
    stand-in's App key where the transport reads it, and every temporary folder made under tmp_path/tmp, so the
    tests can read every file a fetch leaves."""
    mod = _transport_module(monkeypatch, github)
    origin, bare, commits = _owner_app(tmp_path, card)
    monkeypatch.setattr(mod, "GIT_ORIGIN", origin)
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")  # the platform default root, kept in tmp_path too
    root = tmp_path / "root"
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.delenv("HERMES_KANBAN_HOME", raising=False)
    (root / KEY).parent.mkdir(parents=True)
    (root / KEY).write_bytes(github["key_file"].read_bytes())
    _private_temp(monkeypatch, tmp_path)
    return {"mod": mod, "origin": origin, "bare": bare, "commits": commits, "root": root}


def _private_temp(monkeypatch, tmp_path):
    scratch = tmp_path / "tmp"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))


def _refs(git_dir):
    """Every ref the repository holds, with the object each names."""
    listed = _git("--git-dir", str(git_dir), "for-each-ref", "--format=%(refname) %(objectname)")
    return [line.split() for line in listed.splitlines()]


def _objects(git_dir, kind):
    """Every object of *kind* in the repository's store, reachable from a ref or not."""
    listed = _git("--git-dir", str(git_dir), "cat-file", "--batch-all-objects",
                  "--batch-check=%(objecttype) %(objectname)")
    return {name for kind_, name in (line.split() for line in listed.splitlines()) if kind_ == kind}


def _history(git_dir, commit):
    return set(_git("--git-dir", str(git_dir), "rev-list", commit).split())


def _archive_commit(archive):
    """The commit git archive recorded in the archive's own global header, read as git get-tar-commit-id reads it
    (that command itself fails when run from a long working directory, as a worker's task workspace can be)."""
    with tarfile.open(archive) as tar:
        return tar.pax_headers.get("comment")


def _archive_tree(archive, into):
    """The tree git records for the archive's files, unpacked into a fresh repository of their own."""
    _git("init", "-q", str(into))
    with tarfile.open(archive) as tar:
        tar.extractall(into, filter="data")
    _git("add", "-A", "--force", cwd=into)
    return _git("write-tree", cwd=into)


def _blob(git_dir, commit, path):
    """The bytes the repository holds for `path` in `commit`, as they are: no attribute, conversion or decoding."""
    return _RUN(["git", "--git-dir", str(git_dir), "cat-file", "blob", f"{commit}:{path}"], check=True,
                capture_output=True).stdout


def _retar(archive, change):
    """Write the tar at `archive` again, each regular file's bytes passed through `change(name, data)`: None drops
    the file."""
    with tarfile.open(archive) as tar:
        members = [(member, tar.extractfile(member).read() if member.isfile() else None) for member in tar.getmembers()]
        headers = tar.pax_headers
    with tarfile.open(archive, "w", format=tarfile.PAX_FORMAT, pax_headers=headers) as tar:
        for member, data in members:
            if member.isfile():
                data = change(member.name, data)
                if data is None:
                    continue
                member.size = len(data)
            tar.addfile(member, None if data is None else io.BytesIO(data))


def _rewrite_after_archive(monkeypatch, source, change):
    """The module's git runner, with the tar each git archive writes rewritten by `change` right after that git.
    Returns the archives rewritten."""
    run, rewritten = source._run_git, []

    def run_git(argv, env):
        result = run(argv, env)
        if _subcommand(argv) == "archive":
            rewritten.append(next(word.removeprefix("--output=") for word in argv if word.startswith("--output=")))
            _retar(rewritten[-1], change)
        return result

    monkeypatch.setattr(source, "_run_git", run_git)
    return rewritten


# The merge's own .gitattributes asks git archive to leave a tracked file out (export-ignore), to rewrite one
# (export-subst) and to convert two on the way out (text eol=crlf, ident): the archive must still be the merge's tree.
_EXPORT_RULES = {
    ".gitattributes": "ignored.txt export-ignore\nstamp.txt export-subst\ncrlf.txt text eol=crlf\nident.txt ident\n",
    "ignored.txt": "a tracked file that export-ignore would leave out\n",
    "stamp.txt": "built from $Format:%H$\n",
    "crlf.txt": "one\ntwo\n",
    "ident.txt": "$Id$\n",
}

# What could stand between git archive and the build: a file of the tree left out, or one file's bytes changed.
_TAMPERING = {
    "omission": lambda name, data: None if name == "README.md" else data,
    "substitution": lambda name, data: b"# not the reviewed file\n" if name == "README.md" else data,
}


def test_the_copy_holds_exactly_the_asked_merge_and_the_archive_is_its_tree(github, monkeypatch, tmp_path):
    host = _release_host(github, monkeypatch, tmp_path)
    merge, bare = host["commits"]["merge"], host["bare"]
    made = _record_mkdtemp(monkeypatch)

    archive = _source_module().fetch_source(merge)

    copy = _expected_copy(host["root"])
    assert [name for _, name in _refs(copy)] == [merge]  # one ref, on the asked commit: no main, branch or tag
    assert _objects(copy, "commit") == _history(bare, merge)  # the merge's whole history, and no commit beside it
    assert host["commits"]["later"] not in _objects(copy, "commit") and not _objects(copy, "tag")
    assert len(_git("--git-dir", str(copy), "rev-list", "--parents", "-n", "1", merge).split()) == 3  # a merge
    assert _git("--git-dir", str(copy), "rev-parse", "--is-bare-repository", "--is-shallow-repository") == "true\nfalse"
    assert _archive_commit(archive) == merge
    tree = _git("--git-dir", str(bare), "rev-parse", f"{merge}^{{tree}}")  # the merge's tree, on the origin
    assert _archive_tree(archive, tmp_path / "unpacked") == tree
    assert list(archive.parent.iterdir()) == [archive]  # the archive alone, in a private build folder
    assert stat.S_IMODE(archive.parent.stat().st_mode) == 0o700
    assert [folder for folder in made if folder.exists()] == [archive.parent]  # every other private folder is gone


def test_the_archive_is_the_commit_tree_byte_for_byte_whatever_its_own_gitattributes_ask(github, monkeypatch, tmp_path):
    """None of the merge's own export rules and conversions applies. Each file is read straight from the tar, since
    unpacking it into a repository would apply the attributes again and hide a conversion, and holds the bytes of its
    blob on the origin."""
    host = _release_host(github, monkeypatch, tmp_path, _EXPORT_RULES)
    merge, bare = host["commits"]["merge"], host["bare"]
    source, attributes = _source_module(), _expected_copy(host["root"]) / "info" / "attributes"
    source.fetch_source(merge)
    attributes.write_text("* export-ignore\n")  # a rule left in the copy's own attributes file is overwritten

    archive = source.fetch_source(merge)

    faithful = b"* -export-ignore -export-subst -text -eol -crlf -ident -filter -working-tree-encoding\n"
    assert attributes.read_bytes() == faithful and stat.S_IMODE(attributes.parent.stat().st_mode) == 0o700
    with tarfile.open(archive) as tar:
        files = {member.name: tar.extractfile(member).read() for member in tar.getmembers() if member.isfile()}
        links = {member.name: member.linkname for member in tar.getmembers() if member.issym()}
    entries = [line.split(None, 3) for line in _git("--git-dir", str(bare), "ls-tree", "-r", merge).splitlines()]
    assert sorted(files) == sorted(path for mode, _, _, path in entries if mode != "120000")
    assert links == {"app/current.html": "page.html"}
    for path, data in files.items():
        assert data == _blob(bare, merge, path), path  # byte for byte, the blob on the origin
    assert files["ignored.txt"] == _EXPORT_RULES["ignored.txt"].encode()  # export-ignore left nothing out
    assert files["stamp.txt"] == b"built from $Format:%H$\n"  # export-subst rewrote nothing
    assert files["crlf.txt"] == b"one\ntwo\n"  # eol=crlf converted no line end
    assert files["ident.txt"] == b"$Id$\n"  # and ident expanded nothing
    assert _archive_commit(archive) == merge
    plain = tmp_path / "plain.tar"  # the control: git archive run on the origin, where the merge's attributes apply
    _git("--git-dir", str(bare), "archive", "--format=tar", f"--output={plain}", merge)
    with tarfile.open(plain) as tar:
        exported = {member.name: tar.extractfile(member).read() for member in tar.getmembers() if member.isfile()}
    assert "ignored.txt" not in exported
    assert [name for name in ("stamp.txt", "crlf.txt", "ident.txt") if exported[name] == files[name]] == []


@pytest.mark.parametrize("tampering", sorted(_TAMPERING))
def test_an_archive_that_is_not_the_commit_tree_is_refused_and_leaves_no_build_folder(
        github, monkeypatch, tmp_path, tampering):
    """The proof: right after git archive the tar is written again, one file left out or one file's bytes changed.
    The archive is checked against the commit's tree before it is returned, so it is refused and removed."""
    host = _release_host(github, monkeypatch, tmp_path)
    source = _source_module()
    rewritten, made = _rewrite_after_archive(monkeypatch, source, _TAMPERING[tampering]), _record_mkdtemp(monkeypatch)

    with pytest.raises(host["mod"].GitHubTransportError) as refused:
        source.fetch_source(host["commits"]["merge"])

    assert refused.value.reason == "archive_failed"
    assert len(rewritten) == 1 and Path(rewritten[0]).parent in made  # the archive was written, then rewritten
    assert not [folder for folder in made if folder.exists()] and list((tmp_path / "tmp").iterdir()) == []


_NOT_A_COMMIT = [
    ("short", lambda sha: sha[:7]), ("one_short", lambda sha: sha[:39]), ("one_long", lambda sha: sha + "0"),
    ("upper", lambda sha: sha.upper()), ("branch", lambda sha: "main"), ("ref", lambda sha: "refs/heads/main"),
    ("tag", lambda sha: "v1"), ("head", lambda sha: "HEAD"), ("relative", lambda sha: "HEAD~1"),
    ("peeled", lambda sha: sha + "^{commit}"), ("refspec", lambda sha: f"+{sha}:refs/heads/main"),
    ("spaced", lambda sha: f" {sha}"), ("newline", lambda sha: f"{sha}\n"), ("option", lambda sha: "--all"),
    ("empty", lambda sha: ""), ("none", lambda sha: None), ("bytes", lambda sha: sha.encode()),
]


@pytest.mark.parametrize("form", [form for _, form in _NOT_A_COMMIT], ids=[name for name, _ in _NOT_A_COMMIT])
def test_a_short_or_symbolic_commit_is_refused_before_any_token_exists(github, monkeypatch, tmp_path, form):
    host = _release_host(github, monkeypatch, tmp_path)
    mod, given = host["mod"], form(host["commits"]["merge"])
    assert given != host["commits"]["merge"]
    source = _source_module()
    calls, made = _record_git(monkeypatch), _record_mkdtemp(monkeypatch)

    with pytest.raises(mod.GitHubTransportError) as refused:
        source.fetch_source(given)
    assert refused.value.reason == "bad_commit"
    transport = mod.GitHubTransport("release_source", OWNER_APP, key_path=github["key_file"])
    with pytest.raises(mod.GitHubTransportError) as refused:  # the transport's own commit fetch refuses it too
        transport.fetch_commit(tmp_path / "copy.git", given)
    assert refused.value.reason == "bad_commit"

    assert github["token_requests"] == [] and github["jwts"] == [] and github["calls"] == []
    assert calls == [] and made == []  # no git started and no private folder was made
    assert not (host["root"] / "kanban").exists() and not (tmp_path / "copy.git").exists()


def test_no_file_of_the_copy_or_archive_holds_the_token(github, monkeypatch, tmp_path):
    host = _release_host(github, monkeypatch, tmp_path)
    calls, made = _record_git(monkeypatch), _record_mkdtemp(monkeypatch)

    archive = _source_module().fetch_source(host["commits"]["merge"])

    carrying = [(argv, env) for argv, env in calls if _carries(argv, env)]
    assert github["token_requests"] and len(carrying) == 1  # a token was minted and used
    (argv, env), = carrying  # by a git run in a private folder made for it, and gone now
    assert Path(env["HOME"]) in made and argv[argv.index("--git-dir") + 1].startswith(env["HOME"] + "/")
    assert not Path(env["HOME"]).exists()
    secrets = (TOKEN, BASIC, *github["jwts"])
    copy = _expected_copy(host["root"])
    texts = _files_text(tmp_path)  # the copy, the archive, its build folder and every other file under tmp_path
    assert archive in texts and any(path.is_relative_to(copy) for path in texts)
    for path, text in texts.items():
        assert not [secret for secret in secrets if secret in text], path
    with tarfile.open(archive) as tar:  # and every file inside the archive, read as it unpacks
        for member in tar.getmembers():
            data = tar.extractfile(member).read().decode("utf-8", "replace") if member.isfile() else member.linkname
            assert not [secret for secret in secrets if secret in data or secret in member.name], member.name
    config = _git("config", "--file", str(copy / "config"), "--list")
    assert "extraheader" not in config and "credential" not in config and "remote." not in config
    assert [folder for folder in made if folder.exists()] == [archive.parent]


def test_the_token_request_names_only_contents_read_and_only_the_owner_app_repository(github, monkeypatch, tmp_path):
    host = _release_host(github, monkeypatch, tmp_path)
    merge = host["commits"]["merge"]
    calls = _record_git(monkeypatch)

    _source_module().fetch_source(merge)

    assert github["token_requests"] == [OWNER_APP_TOKEN]
    assert len(github["jwts"]) == 1 and github["rejected_jwts"] == 0 and github["calls"] == []  # and no REST call
    carrying = [argv for argv, env in calls if _carries(argv, env)]
    assert len(carrying) == 1 and _subcommand(carrying[0]) == "fetch"  # the token reaches one git, a fetch
    fetch = carrying[0]
    named = [word for word in fetch[fetch.index("fetch") + 1:] if not word.startswith("-")]
    assert named[0] == f"{host['origin']}/{OWNER_APP}.git"  # one repository, the owner app
    assert len(named) == 2 and named[1].removeprefix("+").split(":")[0] == merge  # one refspec: this commit alone
    assert "--no-tags" in fetch and not [word for word in fetch if TOKEN in word or BASIC in word]


@pytest.mark.parametrize("layout", ["production", "custom"])
def test_under_the_dispatcher_worker_environment_the_copy_lands_under_the_kanban_root(
        github, monkeypatch, tmp_path, layout):
    """Run as the dispatcher runs a worker: HERMES_HOME at a profile home under the root, the worker's own board
    pinned in HERMES_KANBAN_DB and HERMES_KANBAN_BOARD, and its task named. The copy still lands under the kanban
    root, whose path is worked out here from the layout, and the App key read is the root's: the profile, the board
    and the workspace each hold another key, which the stand-in refuses."""
    mod = _transport_module(monkeypatch, github)
    origin, bare, commits = _owner_app(tmp_path)
    monkeypatch.setattr(mod, "GIT_ORIGIN", origin)
    _private_temp(monkeypatch, tmp_path)
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: home)  # the platform default root, as test_profiles.py moves it
    # production: the platform default root; custom: a root elsewhere, as in Docker or custom installs
    root = home / ".hermes" if layout == "production" else tmp_path / "custom-root"
    profile = root / "profiles" / "raphael-claude-worker"
    board = root / "kanban" / "boards" / "delivery"
    workspace = board / "workspaces" / "t_0001"
    for path in (profile, workspace):
        path.mkdir(parents=True)
    monkeypatch.delenv("HERMES_KANBAN_HOME", raising=False)
    worker_env = {  # every variable _default_spawn sets, with the worker's own board pinned
        "HERMES_HOME": profile, "HERMES_TENANT": "raphael", "HERMES_KANBAN_TASK": "t_0001",
        "HERMES_KANBAN_WORKSPACE": workspace, "HERMES_SESSION_SOURCE": "kanban", "TERMINAL_CWD": workspace,
        "HERMES_KANBAN_BRANCH": "delivery/t_0001", "HERMES_KANBAN_RUN_ID": "1",
        "HERMES_KANBAN_CLAIM_LOCK": "dispatcher:1", "HERMES_KANBAN_GOAL_MODE": "1",
        "HERMES_KANBAN_GOAL_MAX_TURNS": "20", "TERMINAL_TIMEOUT": "3600", "TERMINAL_MAX_FOREGROUND_TIMEOUT": "3600",
        "HERMES_KANBAN_DB": board / "kanban.db", "HERMES_KANBAN_WORKSPACES_ROOT": board / "workspaces",
        "HERMES_KANBAN_BOARD": "delivery", "HERMES_PROFILE": "raphael-claude-worker",
        "XDG_RUNTIME_DIR": tmp_path / "run", "DBUS_SESSION_BUS_ADDRESS": "unix:path=" + str(tmp_path / "run" / "bus"),
    }
    for name, value in worker_env.items():
        monkeypatch.setenv(name, str(value))
    monkeypatch.chdir(workspace)  # the dispatcher starts the worker inside the task workspace
    root_key = _new_key()
    for base, key in ((root, root_key), (profile, _new_key()), (board, _new_key()), (workspace, _new_key())):
        (base / KEY).parent.mkdir(parents=True)
        (base / KEY).write_bytes(_pem(key))
    github["public"] = root_key.public_key()  # the stand-in accepts a JWT signed with the root key only
    before = {base: sorted(base.rglob("*")) for base in (profile, board, workspace)}

    archive = _source_module().fetch_source(commits["merge"])

    copy = root / "kanban" / "release-source" / "raphael-workspace.git"
    assert [name for _, name in _refs(copy)] == [commits["merge"]]
    assert {base: sorted(base.rglob("*")) for base in before} == before  # nothing in the profile, board or workspace
    assert _archive_commit(archive) == commits["merge"]
    assert len(github["jwts"]) == 1 and github["rejected_jwts"] == 0  # the stand-in's own count
    assert github["token_requests"] == [OWNER_APP_TOKEN]


def test_a_relative_kanban_root_is_anchored_once_at_the_working_directory_of_the_call(github, monkeypatch, tmp_path):
    """HERMES_KANBAN_HOME may be relative, and kanban_home() then is too. The copy is anchored at the working
    directory of the call, so every git, each run in a private folder, names that one copy: none is made inside a
    temporary folder."""
    host = _release_host(github, monkeypatch, tmp_path)
    merge, bare = host["commits"]["merge"], host["bare"]
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(Path("shared") / "root"))
    source, made, inside = _source_module(), _record_mkdtemp(monkeypatch), []

    def run(argv, *args, **kwargs):  # each git, then a look for a copy inside every private folder made so far
        result = _RUN(argv, *args, **kwargs)
        inside.extend(path for folder in made if folder.exists() for path in folder.rglob("raphael-workspace.git"))
        return result

    monkeypatch.setattr(subprocess, "run", run)

    archive = source.fetch_source(merge)

    copy = _expected_copy(tmp_path / "shared" / "root")
    assert source.copy_path() == copy and source.copy_path().is_absolute()
    assert [name for _, name in _refs(copy)] == [merge] and _objects(copy, "commit") == _history(bare, merge)
    assert _archive_commit(archive) == merge
    tree = _git("--git-dir", str(bare), "rev-parse", f"{merge}^{{tree}}")
    assert _archive_tree(archive, tmp_path / "unpacked") == tree
    assert made and inside == []  # no copy inside any temporary folder


def test_the_step_list_holds_release_source_with_exactly_contents_read():
    steps = _module().STEP_PERMISSIONS
    assert "release_source" in steps and dict(steps["release_source"]) == CONTENTS_READ


def test_the_new_step_adds_no_permission_and_the_fetch_no_other_credential(github, monkeypatch, tmp_path):
    steps = _module().STEP_PERMISSIONS
    others = [permissions for step, permissions in steps.items() if step != "release_source"]
    for name, level in steps["release_source"].items():  # each already held by another step (write includes read)
        assert [permissions for permissions in others if permissions.get(name) in (level, "write")], name
    host = _release_host(github, monkeypatch, tmp_path)
    root = host["root"]
    before = sorted(root.rglob("*"))
    assert before == [root / "secrets", root / KEY.parent, root / KEY]  # the App key, alone
    for name in ("GITHUB_TOKEN", "GH_TOKEN"):  # a credential on the host that no git of the fetch may carry
        monkeypatch.setenv(name, DUMMY)
    calls = _record_git(monkeypatch)

    _source_module().fetch_source(host["commits"]["merge"])

    copy = _expected_copy(root)
    added = [path for path in root.rglob("*") if path not in before]
    assert added and all(path.is_relative_to(copy) or copy.is_relative_to(path) for path in added)  # the copy alone
    assert sorted((root / "secrets").rglob("*")) == before[1:]  # no key, token or secret file beside the App key
    assert github["token_requests"] == [OWNER_APP_TOKEN] and github["calls"] == []
    assert calls and not [argv for argv, env in calls if DUMMY in " ".join([*argv, *env.values()])]


def test_only_the_release_source_step_fetches_a_commit(github, monkeypatch, tmp_path):
    host = _release_host(github, monkeypatch, tmp_path)
    mod, merge = host["mod"], host["commits"]["merge"]
    calls, made = _record_git(monkeypatch), _record_mkdtemp(monkeypatch)
    copy = tmp_path / "copy.git"

    for step in mod.STEP_PERMISSIONS:
        if step != "release_source":
            with pytest.raises(mod.GitHubTransportError) as refused:
                mod.GitHubTransport(step, OWNER_APP, key_path=github["key_file"]).fetch_commit(copy, merge)
            assert refused.value.reason == "step_not_permitted", step

    assert github["token_requests"] == [] and calls == [] and made == [] and not copy.exists()


def test_the_release_source_step_fetches_one_commit_and_no_main(github, monkeypatch, tmp_path):
    """One commit, the one asked: main reaches the step neither by fetch_main nor by clone_main, each refused before
    any token, git or folder, though a checkout of the owner app and an empty target are there to take it."""
    host = _release_host(github, monkeypatch, tmp_path)
    mod = host["mod"]
    checkout, target = tmp_path / "checkout", tmp_path / "clone"
    _git("clone", "-q", "-b", "main", str(host["bare"]), str(checkout))
    refs = _refs(checkout / ".git")
    calls, made = _record_git(monkeypatch), _record_mkdtemp(monkeypatch)
    transport = mod.GitHubTransport("release_source", OWNER_APP, key_path=github["key_file"])

    for fetch, folder in ((transport.fetch_main, checkout), (transport.clone_main, target)):
        with pytest.raises(mod.GitHubTransportError) as refused:
            fetch(folder)
        assert refused.value.reason == "step_not_permitted", fetch.__name__

    assert github["token_requests"] == [] and calls == [] and made == [] and not target.exists()
    assert _refs(checkout / ".git") == refs  # no main fetched into the checkout


def test_a_later_fetch_moves_the_one_ref_and_each_archive_has_a_folder_of_its_own(github, monkeypatch, tmp_path):
    host = _release_host(github, monkeypatch, tmp_path)
    merge, later, bare = host["commits"]["merge"], host["commits"]["later"], host["bare"]
    source, copy = _source_module(), _expected_copy(host["root"])

    first, again = source.fetch_source(merge), source.fetch_source(merge)  # the same commit twice
    assert [name for _, name in _refs(copy)] == [merge]
    second = source.fetch_source(later)
    assert [name for _, name in _refs(copy)] == [later]  # the one ref moved, and the merge stays in its history
    assert _objects(copy, "commit") == _history(bare, later) and not _objects(copy, "tag")

    assert [_archive_commit(archive) for archive in (first, again, second)] == [merge, merge, later]
    assert len({first.parent, again.parent, second.parent}) == 3
    assert github["token_requests"] == [OWNER_APP_TOKEN] * 3  # one token for each fetch


def test_a_commit_the_owner_app_does_not_hold_is_refused_and_leaves_no_folder(github, monkeypatch, tmp_path):
    host = _release_host(github, monkeypatch, tmp_path)
    made = _record_mkdtemp(monkeypatch)

    with pytest.raises(host["mod"].GitHubTransportError) as refused:
        _source_module().fetch_source("f" * 40)

    assert refused.value.reason == "fetch_failed"
    assert made and not [folder for folder in made if folder.exists()]  # no private copy and no build folder left
    assert _refs(_expected_copy(host["root"])) == []
    for path, text in _files_text(tmp_path).items():
        assert TOKEN not in text and BASIC not in text, path


def test_one_transport_fetches_one_commit_and_another_commit_needs_a_token_of_its_own(github, monkeypatch, tmp_path):
    """Each credential serves exactly one commit: a release_source transport is bound to the first commit it fetches.
    Another commit on it is refused before any token, git or folder, the bound commit again is served on the same
    token, and the other commit takes a transport and a token request of its own."""
    host = _release_host(github, monkeypatch, tmp_path)
    mod, merge, other = host["mod"], host["commits"]["merge"], host["commits"]["other"]
    copy = tmp_path / "copy.git"
    _git("init", "-q", "--bare", str(copy))
    transport = mod.GitHubTransport("release_source", OWNER_APP, key_path=github["key_file"])
    assert transport.fetch_commit(copy, merge) == merge
    assert github["token_requests"] == [OWNER_APP_TOKEN]
    refs, calls, made = _refs(copy), _record_git(monkeypatch), _record_mkdtemp(monkeypatch)

    with pytest.raises(mod.GitHubTransportError) as refused:
        transport.fetch_commit(copy, other)

    assert refused.value.reason == "step_not_permitted"
    assert github["token_requests"] == [OWNER_APP_TOKEN] and calls == [] and made == []
    assert _refs(copy) == refs == [["refs/heads/fetched", merge]]  # the copy's refs are unchanged
    assert transport.fetch_commit(copy, merge) == merge  # the same credential and the same commit again
    assert github["token_requests"] == [OWNER_APP_TOKEN]
    fresh = mod.GitHubTransport("release_source", OWNER_APP, key_path=github["key_file"])
    assert fresh.fetch_commit(copy, other) == other  # the other commit: a transport and a token request of its own
    assert github["token_requests"] == [OWNER_APP_TOKEN] * 2 and _refs(copy) == [["refs/heads/fetched", other]]


def test_a_transport_whose_first_fetch_failed_stays_bound_to_that_commit(github, monkeypatch, tmp_path):
    """The binding holds though the first fetch failed, the origin not holding that commit: a later fetch of another
    commit on the same transport is refused, with no further token request and no git."""
    host = _release_host(github, monkeypatch, tmp_path)
    mod = host["mod"]
    copy = tmp_path / "copy.git"
    _git("init", "-q", "--bare", str(copy))
    transport = mod.GitHubTransport("release_source", OWNER_APP, key_path=github["key_file"])
    with pytest.raises(mod.GitHubTransportError) as refused:
        transport.fetch_commit(copy, "f" * 40)
    assert refused.value.reason == "fetch_failed"
    requests, calls, made = list(github["token_requests"]), _record_git(monkeypatch), _record_mkdtemp(monkeypatch)

    with pytest.raises(mod.GitHubTransportError) as refused:
        transport.fetch_commit(copy, host["commits"]["merge"])

    assert refused.value.reason == "step_not_permitted"
    assert github["token_requests"] == requests == [OWNER_APP_TOKEN] and calls == [] and made == []
    assert _refs(copy) == []


@pytest.mark.parametrize("repository", [OWNER_APP, PLATFORM])
def test_the_release_source_step_makes_no_rest_call_on_any_repository(github, monkeypatch, repository):
    """The step's one operation is the exact fetch of one owner app commit. A branch head and every allowlisted call,
    each with sample values and its required query, are refused before any token or socket, on the owner app and on
    the platform repository alike."""
    mod = _transport_module(monkeypatch, github)
    opened = _block_sockets(monkeypatch)
    transport = mod.GitHubTransport("release_source", repository, key_path=github["key_file"])

    with pytest.raises(mod.GitHubTransportError) as refused:
        transport.branch_head("card-1")
    assert refused.value.reason == "step_not_permitted"
    for entry in mod.REST_ALLOWLIST:
        with pytest.raises(mod.GitHubTransportError) as refused:
            transport.request(entry.method, entry.template.format(**{**SAMPLE, "repo": repository}),
                              query={k: SAMPLE[k] for k in entry.required_query})
        # Each allowlisted call passes its route and query, then meets the step; the App's own token call is no route.
        assert refused.value.reason == ("step_not_permitted" if entry.permission else "endpoint_not_allowed"), entry

    assert github["token_requests"] == [] and github["jwts"] == [] and github["calls"] == [] and opened == []


def test_the_release_source_step_fetches_no_commit_of_another_repository(github, monkeypatch, tmp_path):
    """Bound to the platform repository, whose copy on the origin holds the same commit, the step's commit fetch is
    refused before any token, private folder or git: it fetches from the owner app alone."""
    host = _release_host(github, monkeypatch, tmp_path)
    mod, merge = host["mod"], host["commits"]["merge"]
    _git("clone", "-q", "--bare", str(host["bare"]), str(host["bare"].parent / "hermes-agent.git"))
    copy = tmp_path / "copy.git"
    _git("init", "-q", "--bare", str(copy))
    calls, made = _record_git(monkeypatch), _record_mkdtemp(monkeypatch)

    with pytest.raises(mod.GitHubTransportError) as refused:
        mod.GitHubTransport("release_source", PLATFORM, key_path=github["key_file"]).fetch_commit(copy, merge)
    assert refused.value.reason == "step_not_permitted"

    assert github["token_requests"] == [] and github["jwts"] == [] and calls == [] and made == []
    assert _refs(copy) == []


def test_the_release_source_step_pushes_nothing(github, monkeypatch, tmp_path):
    """Nor does the step push: a leased push of an owner app commit to its own branch is refused before any token, git
    or folder, though a checkout holding the commit is there."""
    host = _release_host(github, monkeypatch, tmp_path)
    mod, commits = host["mod"], host["commits"]
    checkout = tmp_path / "checkout"
    _git("clone", "-q", "-b", "main", str(host["bare"]), str(checkout))
    refs = _refs(host["bare"])
    calls, made = _record_git(monkeypatch), _record_mkdtemp(monkeypatch)
    transport = mod.GitHubTransport("release_source", OWNER_APP, key_path=github["key_file"])

    with pytest.raises(mod.GitHubTransportError) as refused:
        transport.push(checkout, "card", commits["later"], expected=commits["card"])
    assert refused.value.reason == "step_not_permitted"

    assert github["token_requests"] == [] and calls == [] and made == []
    assert _refs(host["bare"]) == refs  # no branch moved on the origin
