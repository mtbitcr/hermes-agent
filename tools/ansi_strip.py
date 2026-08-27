"""Strip ANSI escape sequences from subprocess output.

Used by terminal_tool, code_execution_tool, and process_registry to clean
command output before returning it to the model.  This prevents ANSI codes
from entering the model's context — which is the root cause of models
copying escape sequences into file writes.

Covers the full ECMA-48 spec: CSI (including private-mode ``?`` prefix,
colon-separated params, intermediate bytes), OSC (BEL and ST terminators),
DCS/SOS/PM/APC string sequences, nF multi-byte escapes, Fp/Fe/Fs
single-byte escapes, and 8-bit C1 control characters.
"""

import re
import unicodedata

_ANSI_ESCAPE_RE = re.compile(
    r"\x1b"
    r"(?:"
        r"\[[\x30-\x3f]*[\x20-\x2f]*[\x40-\x7e]"     # CSI sequence
        r"|\][\s\S]*?(?:\x07|\x1b\\)"                  # OSC (BEL or ST terminator)
        r"|[PX^_][\s\S]*?(?:\x1b\\)"                   # DCS/SOS/PM/APC strings
        r"|[\x20-\x2f]+[\x30-\x7e]"                    # nF escape sequences
        r"|[\x30-\x7e]"                                 # Fp/Fe/Fs single-byte
    r")"
    r"|\x9b[\x30-\x3f]*[\x20-\x2f]*[\x40-\x7e]"       # 8-bit CSI
    r"|\x9d[\s\S]*?(?:\x07|\x9c)"                       # 8-bit OSC
    r"|[\x80-\x9f]",                                    # Other 8-bit C1 controls
    re.DOTALL,
)

# Fast-path check — skip full regex when no escape-like bytes are present.
_HAS_ESCAPE = re.compile(r"[\x1b\x80-\x9f]")

# C0 control characters (minus tab/newline/carriage-return, handled
# separately) plus DEL. These survive strip_ansi() — it only removes
# well-formed escape *sequences* — but are still dangerous or garbled
# when echoed back to a terminal (BEL rings, backspace/DEL overwrite,
# NUL truncates in some terminals).
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Fast-path check for sanitize_display_text — any C0 control (except
# tab/newline), CR, DEL, ESC, or C1 byte triggers the slow path.
_HAS_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")

# Unicode TAG characters (U+E0000–U+E007F).  Deprecated as language tags,
# these render as nothing in every terminal and chat UI but are perfectly
# visible to an LLM tokenizer — the classic "ASCII smuggling" prompt-injection
# channel (hide `\u{E0069}\u{E0067}\u{E006E}...` = invisible instructions
# inside otherwise benign tool output).  Ported from block/goose#10746.
#
# The ONLY legitimate modern use is the three emoji tag sequences Unicode
# actually defines as RGI (TR51): the subdivision flags of England, Scotland
# and Wales — a U+1F3F4 black-flag base, the exact lowercase subdivision code
# as tag characters, and the U+E007F CANCEL TAG terminator.  goose strips
# those too; we preserve exactly those three sequences and nothing else.
#
# Pinning the sequences rather than the SHAPE is the whole point: a U+1F3F4
# followed by an arbitrary tag payload and a CANCEL TAG is not a flag, it is a
# well-formed smuggling frame.  Preserving the shape would leave the entire
# channel open to anything an attacker parks behind one visible black flag.
_TAG_FLAG_BASE = "\U0001F3F4"
_TAG_CANCEL = "\U000E007F"
_PINNED_TAG_FLAGS = tuple(
    _TAG_FLAG_BASE + "".join(chr(0xE0000 + ord(ch)) for ch in code) + _TAG_CANCEL
    for code in ("gbeng", "gbsct", "gbwls")
)

_UNICODE_TAG_SUB_RE = re.compile(
    "(" + "|".join(_PINNED_TAG_FLAGS) + ")"  # the three pinned flags (kept)
    + r"|[\U000E0000-\U000E007F]"            # every other tag char (stripped)
)

# Fast-path check — plane-14 tag chars only.
_HAS_UNICODE_TAG = re.compile(r"[\U000E0000-\U000E007F]")

# Unicode 17.0 Default_Ignorable_Code_Point outside the tag block above.
# These code points normally render as nothing while still splitting tokens,
# identifiers and credential names. Keep the list pinned to the normative
# DerivedCoreProperties.txt ranges so security boundaries do not grow an
# incomplete private subset over time. The tag block is handled separately by
# strip_unicode_tags(), which preserves only the three pinned RGI subdivision
# flags.
_DEFAULT_IGNORABLE_NON_TAG_RE = re.compile(
    "["
    "\u00ad\u034f\u061c"
    "\u115f-\u1160\u17b4-\u17b5\u180b-\u180f"
    "\u200b-\u200f\u202a-\u202e\u2060-\u206f"
    "\u3164\ufe00-\ufe0f\ufeff\uffa0\ufff0-\ufff8"
    "\U0001bca0-\U0001bca3\U0001d173-\U0001d17a"
    "\U000e0080-\U000e0fff"
    "]"
)
_HAS_DEFAULT_IGNORABLE_NON_TAG = _DEFAULT_IGNORABLE_NON_TAG_RE


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from text.

    Returns the input unchanged (fast path) when no ESC or C1 bytes are
    present.  Safe to call on any string — clean text passes through
    with negligible overhead.
    """
    if not text or not _HAS_ESCAPE.search(text):
        return text
    return _ANSI_ESCAPE_RE.sub("", text)


def sanitize_display_text(text: str) -> str:
    """Sanitize stored/untrusted text before echoing it to a terminal.

    Removes ANSI/ECMA-48 escape sequences AND bare control characters,
    preserving only newlines and tabs (carriage returns are normalized
    to newlines so ``\\r``-overwrite spoofing can't hide content).

    Use this when re-rendering conversation history or other persisted
    text in a terminal UI (e.g. the ``/resume`` recap): a message that
    arrived with embedded escapes — pasted content, gateway-origin
    text, or model output echoing injected tool results — must not be
    able to clear the screen, retitle the window, move the cursor, or
    restyle adjacent UI when replayed. Rich's ``Text()`` does NOT
    neutralize raw escape bytes, so sanitization has to happen before
    display. Mirrors openai/codex#31494 (``sanitize_user_text``).
    """
    if not text or not _HAS_CONTROL.search(text):
        return text
    text = strip_ansi(text)
    if "\r" in text:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
    return _CONTROL_CHARS_RE.sub("", text)


def strip_unicode_tags(text: str) -> str:
    """Remove invisible Unicode TAG characters (U+E0000–U+E007F) from text.

    Tag characters are invisible in terminals and chat UIs but fully visible
    to LLM tokenizers, making them a prompt-injection smuggling channel for
    untrusted tool output (MCP servers, web content).  The only sequences
    preserved are the three pinned RGI subdivision flags — England
    (``gbeng``), Scotland (``gbsct``) and Wales (``gbwls``), each a U+1F3F4
    base + those exact lowercase tag letters + U+E007F CANCEL TAG.  Every
    other tag character is stripped, including a payload wrapped in the same
    base/CANCEL frame, an orphan tag with no base, and an unterminated
    sequence (whose visible U+1F3F4 base survives on its own).

    Plane 14 is the entire scope: ZWJ (U+200D) and the other invisible/bidi
    characters are deliberately left untouched here, because they carry
    meaning inside legitimate emoji and RTL text.  A caller that needs full
    display hardening filters ``tools.threat_patterns.INVISIBLE_CHARS`` on
    top of this, plus whatever its own boundary removes — see
    ``hermes_cli.owner_workspace._owner_display_text``.

    Returns the input unchanged (fast path) when no plane-14 tag characters
    are present.  Ported from block/goose#10746.
    """
    if not text or not _HAS_UNICODE_TAG.search(text):
        return text
    return _UNICODE_TAG_SUB_RE.sub(lambda m: m.group(1) or "", text)


def is_contextual_zwnj(text: str, index: int) -> bool:
    """Return whether U+200C joins two Arabic-script letters.

    Persian and Urdu use ZWNJ as real orthography.  It is retained only in
    that narrow context; between Latin letters or at a boundary it remains an
    invisible token-splitting character and is removed.
    """
    if index < 0 or index >= len(text) or text[index] != "\u200c":
        return False

    def _base(step: int) -> str:
        cursor = index + step
        while 0 <= cursor < len(text):
            char = text[cursor]
            if not unicodedata.category(char).startswith("M"):
                return char
            cursor += step
        return ""

    left = _base(-1)
    right = _base(1)
    return bool(
        left
        and right
        and unicodedata.category(left).startswith("L")
        and unicodedata.category(right).startswith("L")
        and unicodedata.bidirectional(left) == "AL"
        and unicodedata.bidirectional(right) == "AL"
    )


def strip_default_ignorables(text: str) -> str:
    """Remove Unicode default-ignorable code points except pinned tag flags.

    Callers use this before matching security-sensitive visible text. It first
    removes non-RGI tag characters, then removes every other code point in the
    Unicode 17.0 Default_Ignorable_Code_Point property. The three pinned RGI
    subdivision flags remain intact because they are visible emoji sequences.
    """
    if not text:
        return text
    text = strip_unicode_tags(text)
    if not _HAS_DEFAULT_IGNORABLE_NON_TAG.search(text):
        return text
    return _DEFAULT_IGNORABLE_NON_TAG_RE.sub(
        lambda match: (
            "\u200c"
            if match.group(0) == "\u200c" and is_contextual_zwnj(text, match.start())
            else ""
        ),
        text,
    )
