from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
import http.client
import http.server
import importlib.util
import io
import json
import os
from pathlib import Path
import ssl
import time
import types

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.x509.oid import NameOID

PROXY_PATH = Path(__file__).resolve().parents[2] / "scripts" / "journeys" / "synthetic_proxy.py"
FROZEN_NOW = 1_700_000_000


class _InertServer:
    def __init__(self, *_args, **_kwargs):
        self.socket = object()

    def serve_forever(self):
        return None


class _InertContext:
    def __init__(self, *_args, **_kwargs):
        pass

    def load_cert_chain(self, *_args, **_kwargs):
        return None

    def wrap_socket(self, sock, **_kwargs):
        return sock


def proxy_source_path() -> Path:
    old = os.environ.get("SYNTHETIC_OLD_PROXY")
    if old and Path(old).is_file():
        return Path(old)
    return PROXY_PATH


def load_proxy(monkeypatch, state: Path):
    """Exec the proxy source with server and TLS pieces replaced by inert stand-ins."""
    monkeypatch.setenv("SYNTHETIC_STATE", str(state))
    path = proxy_source_path()
    module = types.ModuleType("synthetic_proxy_under_test")
    module.__file__ = str(path)
    code = compile(path.read_text(), str(path), "exec")
    with monkeypatch.context() as patch:
        patch.setattr(http.server, "ThreadingHTTPServer", _InertServer)
        patch.setattr(ssl, "SSLContext", _InertContext)
        exec(code, module.__dict__)
    return module


def start_proxy(monkeypatch, state: Path):
    """Start the proxy once. The old script starts at import; the new one in main()."""
    module = load_proxy(monkeypatch, state)
    if hasattr(module, "main"):
        module.write_material()
    return module


def mint(module) -> str:
    if hasattr(module, "mint_assertion"):
        return module.mint_assertion()
    return module.TOKEN


def b64d(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def verify_assertion(token: str, jwk_set: dict) -> dict:
    header_b64, payload_b64, signature_b64 = token.split(".")
    header = json.loads(b64d(header_b64))
    assert header == {"alg": "RS256", "kid": "raphael-server2-key", "typ": "JWT"}
    jwk = next(key for key in jwk_set["keys"] if key["kid"] == header["kid"])
    public_key = rsa.RSAPublicNumbers(
        int.from_bytes(b64d(jwk["e"]), "big"), int.from_bytes(b64d(jwk["n"]), "big")
    ).public_key()
    public_key.verify(
        b64d(signature_b64),
        f"{header_b64}.{payload_b64}".encode(),
        padding.PKCS1v15(),
        hashes.SHA256(),
    )
    return json.loads(b64d(payload_b64))


def run_handler(module, raw_request: bytes) -> bytes:
    class FakeSocketHandler(module.Handler):
        def setup(self):
            self.rfile = io.BytesIO(raw_request)
            self.wfile = io.BytesIO()

        def finish(self):
            pass

    handler = FakeSocketHandler(None, ("127.0.0.1", 0), None)
    return handler.wfile.getvalue()


class FakeConnection:
    recorded: list[dict] = []

    def __init__(self, host, port, timeout=None):
        self.host, self.port = host, port

    def request(self, method, path, body, headers):
        type(self).recorded.append({"host": self.host, "port": self.port, "headers": dict(headers)})

    def getresponse(self):
        return types.SimpleNamespace(
            status=200,
            read=lambda: b"ok",
            getheaders=lambda: [("Content-Type", "text/plain"), ("Transfer-Encoding", "chunked")],
        )

    def close(self):
        pass


def file_mode(path: Path) -> int:
    return path.stat().st_mode & 0o777


def test_second_start_reuses_authority_and_keys(monkeypatch, tmp_path):
    first = start_proxy(monkeypatch, tmp_path)
    first_ca = (tmp_path / "ca.pem").read_bytes()
    first_jwk = dict(first.JWK)
    names = ("ca-key.pem", "signing-key.pem")
    first_keys = {name: (tmp_path / name).read_bytes() for name in names if (tmp_path / name).exists()}

    second = start_proxy(monkeypatch, tmp_path)

    assert (tmp_path / "ca.pem").read_bytes() == first_ca
    assert second.JWK == first_jwk
    assert second.CA_PEM == first_ca
    assert sorted(first_keys) == sorted(names)
    for name in names:
        assert (tmp_path / name).read_bytes() == first_keys[name], name
    assert file_mode(tmp_path / "ca.pem") == 0o644
    for name in ("ca.pem", "ca-key.pem", "signing-key.pem", "server.pem", "server-key.pem"):
        assert (tmp_path / name).is_file()


def test_assertion_fields_expiry_and_signature(monkeypatch, tmp_path):
    module = start_proxy(monkeypatch, tmp_path)
    before = int(time.time())

    payload = verify_assertion(mint(module), {"keys": [module.JWK]})

    assert payload["iss"] == "https://synthetic-team.cloudflareaccess.com"
    assert payload["aud"] == "raphael-server2-browser"
    assert payload["sub"] == "raphael-server2-owner"
    assert payload["type"] == "app"
    assert payload["email"] == "owner@server2.invalid"
    assert before - 5 <= payload["iat"] <= int(time.time()) + 5
    assert payload["nbf"] == payload["iat"] - 30
    assert payload["exp"] - payload["iat"] == 600
    assert set(payload) - {"iss", "aud", "sub", "type", "email", "iat", "nbf", "exp"} <= {"jti"}


def test_two_requests_get_separately_minted_assertions(monkeypatch, tmp_path):
    module = start_proxy(monkeypatch, tmp_path)
    FakeConnection.recorded = []
    monkeypatch.setattr(http.client, "HTTPConnection", FakeConnection)
    # Freeze the clock so both tokens are issued within the same second.
    monkeypatch.setattr(module, "time", types.SimpleNamespace(time=lambda: float(FROZEN_NOW)))

    request = (
        b"GET /api/me HTTP/1.1\r\nHost: workspace.synthetic.test\r\n"
        b"Connection: close\r\nCf-Access-Jwt-Assertion: spoofed\r\n\r\n"
    )
    first_response = run_handler(module, request)
    run_handler(module, request)

    assert first_response.startswith(b"HTTP/1.1 200")
    assert b"chunked" not in first_response
    assert len(FakeConnection.recorded) == 2
    assert FakeConnection.recorded[0]["host"] == "workspace"
    assert FakeConnection.recorded[0]["port"] == 3000
    sent = [call["headers"] for call in FakeConnection.recorded]
    for headers in sent:
        assert headers["host"] == "workspace.synthetic.test"
        assert headers["x-forwarded-proto"] == "https"
        assert not any(key.lower() == "connection" for key in headers)
        assert headers["cf-access-jwt-assertion"] != "spoofed"
    tokens = [headers["cf-access-jwt-assertion"] for headers in sent]
    assert tokens[0] != tokens[1]

    keys_response = run_handler(
        module,
        b"GET /cdn-cgi/access/certs HTTP/1.1\r\n"
        b"Host: synthetic-team.cloudflareaccess.com\r\nConnection: close\r\n\r\n",
    )
    jwk_set = json.loads(keys_response.split(b"\r\n\r\n", 1)[1])
    for token in tokens:
        payload = verify_assertion(token, jwk_set)
        assert payload["iat"] == FROZEN_NOW
        assert payload["exp"] == FROZEN_NOW + 600


def test_private_keys_not_readable_by_others(monkeypatch, tmp_path):
    old_umask = os.umask(0o000)
    try:
        module = start_proxy(monkeypatch, tmp_path)
        for path in sorted(tmp_path.glob("*-key.pem")):
            assert file_mode(path) & 0o077 == 0, path.name
        for name in ("ca-key.pem", "signing-key.pem"):
            assert (tmp_path / name).is_file(), name
            assert file_mode(tmp_path / name) & 0o077 == 0, name
        # A key left loose by something else is tightened on the next start.
        (tmp_path / "signing-key.pem").chmod(0o644)
        (tmp_path / "ca-key.pem").chmod(0o664)
        module.write_material()
        for name in ("ca-key.pem", "signing-key.pem", "server-key.pem"):
            assert file_mode(tmp_path / name) == 0o600, name
    finally:
        os.umask(old_umask)


def _write_old_proxy_state(state: Path) -> None:
    """State as the previous proxy left it: ca.pem, server.pem, server-key.pem (no CA key)."""
    now = datetime.now(timezone.utc)
    ca_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    server_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Raphael Server 2 Test CA")])
    server_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "workspace.synthetic.test")])

    def build(subject, public_key, days, ca):
        builder = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(ca_name)
            .public_key(public_key)
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(minutes=1))
            .not_valid_after(now + timedelta(days=days))
        )
        if ca:
            builder = builder.add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        return builder.sign(ca_key, hashes.SHA256())

    pem = serialization.Encoding.PEM
    files = {
        "ca.pem": build(ca_name, ca_key.public_key(), 1, True).public_bytes(pem),
        "server.pem": build(server_name, server_key.public_key(), 1, False).public_bytes(pem),
        "server-key.pem": server_key.private_bytes(
            pem, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
        ),
    }
    for name, data in files.items():
        (state / name).write_bytes(data)
        (state / name).chmod(0o444)


def test_interrupted_migration_from_old_state_recovers_on_restart(monkeypatch, tmp_path):
    _write_old_proxy_state(tmp_path)
    module = load_proxy(monkeypatch, tmp_path)

    real_replace = os.replace
    calls = []

    def replace_then_die(src, dst):
        calls.append(Path(dst).name)
        if len(calls) == 2:
            raise OSError("killed between CA key write and CA certificate write")
        real_replace(src, dst)

    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", replace_then_die)
        try:
            module.write_material()
        except OSError:
            pass
        else:
            raise AssertionError("migration was not interrupted")
    assert calls[:2] == ["ca-key.pem", "ca.pem"]
    assert (tmp_path / "ca-key.pem").is_file()

    restarted = start_proxy(monkeypatch, tmp_path)

    ca_cert = x509.load_pem_x509_certificate((tmp_path / "ca.pem").read_bytes())
    ca_key = serialization.load_pem_private_key((tmp_path / "ca-key.pem").read_bytes(), None)
    server_cert = x509.load_pem_x509_certificate((tmp_path / "server.pem").read_bytes())
    server_key = serialization.load_pem_private_key((tmp_path / "server-key.pem").read_bytes(), None)
    assert ca_cert.public_key().public_numbers() == ca_key.public_key().public_numbers()
    assert server_cert.public_key().public_numbers() == server_key.public_key().public_numbers()
    server_cert.verify_directly_issued_by(ca_cert)
    assert restarted.CA_PEM == (tmp_path / "ca.pem").read_bytes()

    # A further restart keeps the recovered authority.
    recovered_ca = (tmp_path / "ca.pem").read_bytes()
    start_proxy(monkeypatch, tmp_path)
    assert (tmp_path / "ca.pem").read_bytes() == recovered_ca


def test_server_certificate_is_one_year_and_reissued_near_expiry(monkeypatch, tmp_path):
    module = start_proxy(monkeypatch, tmp_path)
    ca_cert = x509.load_pem_x509_certificate((tmp_path / "ca.pem").read_bytes())
    server_pem = (tmp_path / "server.pem").read_bytes()
    server_cert = x509.load_pem_x509_certificate(server_pem)
    now = datetime.now(timezone.utc)

    lifetime = server_cert.not_valid_after_utc - server_cert.not_valid_before_utc
    assert lifetime == timedelta(days=365, minutes=1)
    assert abs(server_cert.not_valid_after_utc - (now + timedelta(days=365))) < timedelta(minutes=5)
    assert ca_cert.not_valid_after_utc > server_cert.not_valid_after_utc + timedelta(days=365)
    server_cert.verify_directly_issued_by(ca_cert)

    module.write_material()
    assert (tmp_path / "server.pem").read_bytes() == server_pem

    # Replace the certificate with one ending in 10 days, signed by the same authority.
    ca_key = serialization.load_pem_private_key((tmp_path / "ca-key.pem").read_bytes(), None)
    server_key = serialization.load_pem_private_key((tmp_path / "server-key.pem").read_bytes(), None)
    short = (
        x509.CertificateBuilder()
        .subject_name(server_cert.subject)
        .issuer_name(ca_cert.subject)
        .public_key(server_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(days=300))
        .not_valid_after(now + timedelta(days=10))
        .sign(ca_key, hashes.SHA256())
    )
    short_pem = short.public_bytes(serialization.Encoding.PEM)
    (tmp_path / "server.pem").write_bytes(short_pem)

    module.write_material()

    renewed = x509.load_pem_x509_certificate((tmp_path / "server.pem").read_bytes())
    assert (tmp_path / "server.pem").read_bytes() != short_pem
    assert renewed.not_valid_after_utc - now > timedelta(days=364)
    renewed.verify_directly_issued_by(ca_cert)
    sans = renewed.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    assert sorted(sans.get_values_for_type(x509.DNSName)) == [
        "synthetic-team.cloudflareaccess.com",
        "workspace.synthetic.test",
    ]
