"""Shared helpers, fakes and constants of the journeys stack isolation tests (card P-2).

The stand-ins are in journeys_stack_harness; the three test_journeys_stack_isolation*.py files split the
isolation_check.sh tests by run time and import what they use from here.
"""

from __future__ import annotations

import base64
import datetime
import http.server
import ipaddress
import re
import ssl
import threading
from pathlib import Path
from typing import NamedTuple

import pytest

from journeys_stack_harness import (
    APP_C,
    BROWSER_C,
    DECOY_VOLUMES,
    GATE_C,
    Host,
    PROXY_SOURCE,
    SAME_HOST_REDIRECT,
    cron_job,
    dispatcher_on,
    dispatcher_unset,
    hermes_gives,
    platform_enabled,
    profile_folder,
)


def proxy_issuer() -> str:
    match = re.search(r'^ISSUER = "([^"]*)"$', PROXY_SOURCE.read_text(), re.M)
    assert match, "synthetic_proxy.py no longer has an ISSUER constant line"
    return match.group(1)


def b64d(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def add_env_line(host: Host, profile: str, line: str) -> None:
    with open(profile_folder(host, profile) / ".env", "a") as handle:
        handle.write(line + "\n")


def extra_env_name(who: str):
    return lambda host: host.update(lambda s: s["containers"][who]["env"].append("STRAY_SETTING=1"))


def extra_mount(who: str, entry: dict):
    return lambda host: host.update(lambda s: s["containers"][who]["mounts"].append(entry))


def curl_gives(**settings):
    return lambda host: host.update(lambda s: s["curl"].update(settings))


# One thing broken per case: (the check that must fail, how to break it, script environment changes).
BREAKS = {
    "extra-env-name-in-app": ("names-on-template", extra_env_name(APP_C), {}),
    "extra-env-name-in-browser": ("names-on-template", extra_env_name(BROWSER_C), {}),
    "extra-env-name-in-gateway": ("names-on-template", extra_env_name(GATE_C), {}),
    "bind-mount": ("names-on-template",
                   extra_mount(APP_C, {"type": "bind", "name": "", "dest": "/host-dir"}), {}),
    "foreign-volume": ("names-on-template",
                       extra_mount(GATE_C, {"type": "volume", "name": DECOY_VOLUMES[0], "dest": "/old"}), {}),
    "home-env-name-default": ("names-on-template", lambda h: add_env_line(h, "default", "STRAY_SETTING=1"), {}),
    "home-env-name-planner": ("names-on-template", lambda h: add_env_line(h, "planner", "STRAY_SETTING=1"), {}),
    "home-env-name-designer": ("names-on-template", lambda h: add_env_line(h, "designer", "STRAY_SETTING=1"), {}),
    "home-env-line-without-name": ("names-on-template", lambda h: add_env_line(h, "planner", "just-text"), {}),
    "home-env-line-without-name-designer": (
        "names-on-template", lambda h: add_env_line(h, "designer", "just-text"), {}),
    "dispatcher-on-default": ("dispatcher-off", dispatcher_on("default"), {}),
    "dispatcher-on-planner": ("dispatcher-off", dispatcher_on("planner"), {}),
    "dispatcher-on-designer": ("dispatcher-off", dispatcher_on("designer"), {}),
    "dispatcher-unset-planner": ("dispatcher-off", dispatcher_unset("planner"), {}),
    "dispatcher-unset-designer": ("dispatcher-off", dispatcher_unset("designer"), {}),
    "cron-job-default": ("cron-lists-empty", cron_job("default"), {}),
    "cron-job-planner": ("cron-lists-empty", cron_job("planner"), {}),
    "cron-job-designer": ("cron-lists-empty", cron_job("designer"), {}),
    "cron-command-fails": ("cron-lists-empty", hermes_gives(cron_rc=1), {}),
    "platform-in-default-config": ("no-messaging-platform", platform_enabled("default"), {}),
    "platform-in-planner-config": ("no-messaging-platform", platform_enabled("planner"), {}),
    "platform-in-designer-config": ("no-messaging-platform", platform_enabled("designer"), {}),
    "message-delivered": ("telegram-send-fails", hermes_gives(send_rc=0), {}),
    "production-answers-200": ("production-get-refused", curl_gives(code=200), {}),
    "production-answers-404": ("production-get-refused", curl_gives(code=404), {}),
    "production-answers-500": ("production-get-refused", curl_gives(code=500), {}),
    "redirect-to-the-same-host": ("production-get-refused",
                                  curl_gives(code=302, redirect=SAME_HOST_REDIRECT), {}),
    "redirect-to-the-same-host-other-case-port-user": (
        "production-get-refused",
        curl_gives(code=302, redirect="https://someone@PROD.example.invalid:8443/x"), {}),
    "redirect-to-the-same-host-trailing-dot": (
        "production-get-refused", curl_gives(code=302, redirect="https://prod.example.invalid./x"), {}),
    "redirect-without-location": ("production-get-refused", curl_gives(code=302, redirect=""), {}),
    "redirect-without-a-host": ("production-get-refused", curl_gives(code=302, redirect="/login"), {}),
    "curl-fails": ("production-get-refused", curl_gives(rc=28), {}),
    "origin-unset": ("production-get-refused", None, {"JOURNEY_PRODUCTION_ORIGIN": None}),
    "origin-empty": ("production-get-refused", None, {"JOURNEY_PRODUCTION_ORIGIN": ""}),
}


class Seen(NamedTuple):
    method: str
    assertion: str | None
    body_bytes: int


class Refusing(http.server.BaseHTTPRequestHandler):
    """Answers every request, whatever its method, with 403 and no body, and records what arrived."""

    def __getattr__(self, name):
        if name.startswith("do_"):
            return self.refuse
        raise AttributeError(name)

    def refuse(self):
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        self.server.seen.append(Seen(self.command, self.headers.get("Cf-Access-Jwt-Assertion"), len(body)))
        self.send_response(403)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *args):
        pass


class LocalHttps(NamedTuple):
    origin: str
    ca_bundle: Path
    seen: list[Seen]


def write_local_certificates(folder: Path) -> tuple[Path, Path, Path]:
    """A throw-away CA and a server certificate for 127.0.0.1 and localhost; returns the PEM files (ca, cert, key)."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    ca_key, server_key = ec.generate_private_key(ec.SECP256R1()), ec.generate_private_key(ec.SECP256R1())
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "journeys isolation test CA")])
    server_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    now = datetime.datetime.now(datetime.timezone.utc)

    def issue(subject, key, *extensions):
        builder = (
            x509.CertificateBuilder().subject_name(subject).issuer_name(ca_name)
            .public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(hours=1)).not_valid_after(now + datetime.timedelta(days=1))
        )
        for extension, critical in extensions:
            builder = builder.add_extension(extension, critical)
        return builder.sign(ca_key, hashes.SHA256())

    ca_cert = issue(ca_name, ca_key, (x509.BasicConstraints(ca=True, path_length=None), True))
    server_cert = issue(
        server_name, server_key,
        (x509.BasicConstraints(ca=False, path_length=None), True),
        (x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address("127.0.0.1")), x509.DNSName("localhost")]),
         False),
    )
    folder.mkdir()
    pem = serialization.Encoding.PEM
    key_pem = server_key.private_bytes(pem, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    files = folder / "ca.pem", folder / "server.pem", folder / "server-key.pem"
    for path, data in zip(files, (ca_cert.public_bytes(pem), server_cert.public_bytes(pem), key_pem)):
        path.write_bytes(data)
    return files


@pytest.fixture
def local_https(tmp_path: Path):
    """The endpoint in a thread of the test process, on an ephemeral port, trusted through its own CA."""
    ca, cert, key = write_local_certificates(tmp_path / "tls")
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert, key)
    server = http.server.HTTPServer(("127.0.0.1", 0), Refusing)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    server.seen = []
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    try:
        yield LocalHttps(f"https://127.0.0.1:{server.server_address[1]}", ca, server.seen)
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
