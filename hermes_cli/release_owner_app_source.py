"""The owner app's merge commit on the release host, fetched and archived (owner app release plan, card 2).

The release builds the owner app's image from one exact merge commit of mtbitcr/raphael-workspace, on the release
host itself. fetch_source fetches that commit into the platform's own bare copy of the repository, under the kanban
root that every profile and board shares, through the transport's release_source step: the existing GitHub App's
one read-only step, whose token holds contents read on this one repository. As in fetch_main, the credential lives
only in the environment of the one git that fetches, in a fresh private repository removed after it, so none reaches
the copy, a file or a log. The commit is then written as a git archive into a fresh private build folder, the
build's input. That archive is the commit's tree exactly: no export rule or checkout conversion of the commit applies,
and the archive is checked against the tree before it is returned. A tree that an archive cannot represent exactly,
one with a submodule or an empty folder, is refused.

Nothing calls this module yet: the release run starts it at the owner app's build (card 9). Every import is made at
load, and no secret, host address or key file's content is copied, printed or logged.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import tarfile
import tempfile
from pathlib import Path

from hermes_cli.kanban_db import kanban_home
from hermes_cli.kanban_delivery_github import (
    _COMMIT_REPOSITORY, GitHubTransport, GitHubTransportError, _git_base_env, _is_sha, _run_git,
)

REPOSITORY = _COMMIT_REPOSITORY  # the owner app, the one repository this module reads: the transport's own binding
STEP = "release_source"  # the transport's read-only step: contents read and nothing else
# The copy's own attributes file, which outranks every .gitattributes of the commit and the system and global files:
# no export rule and no checkout conversion applies, so git archive writes each file as the commit's blob holds it.
_FAITHFUL = b"* -export-ignore -export-subst -text -eol -crlf -ident -filter -working-tree-encoding\n"
# The member git archive writes for each mode of the commit's tree: a regular file, executable or not, or a symlink.
_MEMBERS = {b"100644": "file", b"100755": "executable", b"120000": "symlink"}


def copy_path() -> Path:
    """The platform's bare copy of the owner app: under the kanban root, never a profile home or a board, so a
    release started from any of them finds the one copy. Resolved through kanban_home, never kanban_db_path(): that
    honours HERMES_KANBAN_DB, which the dispatcher pins to a worker's own board. Absolute: a relative kanban root is
    anchored at the working directory of the call, since every git runs in a private folder of its own. And resolved
    by the filesystem, where Kanban's own files land: a symlink followed by ".." leads to its target's parent."""
    return Path(os.path.realpath(kanban_home())) / "kanban" / "release-source" / "raphael-workspace.git"


def _tree_id(listing: list[tuple[bytes, bytes, bytes]]) -> bytes | None:
    """The id, as git prints it, of the Git tree built from `listing`, each a file's or symlink's path, mode and raw
    20-byte id: a tree object for each folder, its entries in git's order, where a folder sorts as its name and "/",
    and no empty folder. None when a path is listed twice, or both as a file and as a folder."""
    folders = {b"": {}}  # each folder's path, and its entries by name: a mode and raw id, or 40000 and a folder's path
    for path, mode, oid in listing:
        while True:
            folder, _, name = path.rpartition(b"/")
            known = folder in folders
            entries = folders.setdefault(folder, {})
            if name in entries:
                return None  # a path twice, or both a file and a folder
            entries[name] = (mode, oid)
            if known:
                break
            path, mode, oid = folder, b"40000", folder  # a folder met for the first time, into its own folder
    ids = {}
    for folder in sorted(folders, key=len, reverse=True):  # each folder after the folders in it
        entries = sorted(folders[folder].items(),
                         key=lambda item: item[0] + b"/" if item[1][0] == b"40000" else item[0])
        body = b"".join(mode + b" " + name + b"\0" + (ids[oid] if mode == b"40000" else oid)
                        for name, (mode, oid) in entries)
        ids[folder] = hashlib.sha1(b"tree %d\0" % len(body) + body).digest()
    return ids[b""].hex().encode("ascii")


def _is_tree(archive: str, copy: Path, commit: str, env: dict[str, str]) -> bool:
    """Whether the tar file `archive` is exactly the tree of `commit`, as the copy lists it with no credential: each
    blob a regular file at its path with the blob's bytes and executable bit, each symlink a symlink whose target is
    its blob, every path once, no other file, symlink or member of another kind, and the Git tree rebuilt from these
    files and symlinks the commit's own: so no submodule, empty folder or other shape the archive cannot carry."""
    listed = _run_git(["git", "--git-dir", str(copy), "ls-tree", "-r", "-z", "--full-tree", commit], env)
    if listed is None or listed.returncode != 0:
        return False
    try:
        tree = {}
        listing = []  # each listed path, mode and raw id, as they are, for the tree they rebuild
        for record in listed.stdout.split(b"\0")[:-1]:
            entry, path = record.split(b"\t", 1)
            mode, _, blob = entry.split(b" ")
            if mode not in _MEMBERS:
                return False
            tree[path.decode("utf-8", "surrogateescape")] = (_MEMBERS[mode], blob.decode("ascii"))
            listing.append((path, mode, bytes.fromhex(blob.decode("ascii"))))
        seen = set()
        with tarfile.open(archive, encoding="utf-8", errors="surrogateescape") as tar:
            for member in tar.getmembers():
                if member.name in seen:
                    return False
                seen.add(member.name)
                kind, blob = tree.pop(member.name, (None, None))
                if member.isdir():
                    if kind is not None:  # a folder, where the tree holds no file or symlink
                        return False
                    continue
                if member.isfile() and kind == ("executable" if member.mode & 0o111 else "file"):
                    digest = hashlib.sha1(b"blob %d\0" % member.size)
                    with tar.extractfile(member) as data:
                        while chunk := data.read(1 << 20):
                            digest.update(chunk)
                elif member.issym() and kind == "symlink":
                    target = member.linkname.encode("utf-8", "surrogateescape")
                    digest = hashlib.sha1(b"blob %d\0" % len(target) + target)
                else:
                    return False  # a file, symlink or other member that the tree does not hold at that path
                if digest.hexdigest() != blob:
                    return False
        # Unpacked, the archive gives git these files and symlinks, and git records no empty folder: the tree git
        # builds from them must be the commit's own.
        own = _run_git(["git", "--git-dir", str(copy), "rev-parse", "--verify", f"{commit}^{{tree}}"], env)
        if own is None or own.returncode != 0 or own.stdout.strip() != _tree_id(listing):
            return False
    except (OSError, EOFError, ValueError, tarfile.TarError):
        return False
    return not tree  # and no path of the tree left out


def fetch_source(commit: str) -> Path:
    """Fetch the owner app's `commit`, 40 lowercase hex, into the copy, and return the git archive of it: a tar file
    alone in a fresh private build folder, which the caller removes when the build is done. A short, symbolic or
    relative commit is refused (bad_commit) before any token, folder or git exists. An archive that is not the
    commit's tree is removed and refused (archive_failed)."""
    if not _is_sha(commit):
        raise GitHubTransportError("bad_commit")
    transport = GitHubTransport(STEP, REPOSITORY)  # the App's key under the platform root, never a profile's
    copy = copy_path()  # absolute, and the one copy every git below names
    try:
        copy.parent.mkdir(mode=0o700, parents=True, exist_ok=True)  # the owner app's source is private
        home = os.path.abspath(tempfile.mkdtemp(prefix="hermes-release-source-"))  # mode 0700, for this module's gits
    except OSError:
        raise GitHubTransportError("copy_failed") from None
    try:
        env = _git_base_env(home)
        try:
            Path(env["GIT_CONFIG_GLOBAL"]).touch(exist_ok=False)  # the global configuration: empty
        except OSError:
            raise GitHubTransportError("copy_failed") from None
        # Made on first use with an empty --template=, so it holds what init writes; a copy made before is kept.
        made = _run_git(["git", "init", "--bare", "--quiet", "--template=", str(copy)], env)
        if made is None or made.returncode != 0:
            raise GitHubTransportError("copy_failed")
        transport.fetch_commit(copy, commit)  # the only git with the credential, in the transport's own folder
        try:
            (copy / "info").mkdir(mode=0o700, exist_ok=True)
            (copy / "info" / "attributes").write_bytes(_FAITHFUL)  # this one line, whatever the file held before
            build = os.path.abspath(tempfile.mkdtemp(prefix="hermes-release-build-"))  # mode 0700: the input is private
        except OSError:
            raise GitHubTransportError("archive_failed") from None
        archive = os.path.join(build, f"{commit}.tar")
        written = _run_git(["git", "--git-dir", str(copy), "archive", "--format=tar", f"--output={archive}", commit],
                           env)
        if written is None or written.returncode != 0 or not _is_tree(archive, copy, commit, env):
            shutil.rmtree(build, ignore_errors=True)
            raise GitHubTransportError("archive_failed")
    finally:
        shutil.rmtree(home, ignore_errors=True)  # the empty configuration this module's gits read
    return Path(archive)
