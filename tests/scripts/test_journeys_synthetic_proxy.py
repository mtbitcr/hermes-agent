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
import types

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

PROXY_PATH = Path(__file__).resolve().parents[2] / "scripts" / "journeys" / "synthetic_proxy.py"
FROZEN_NOW = 1_700_000_000


class _NoServer:
    def __init__(self, *_args, **_kwargs):
        raise AssertionError("proxy created a server at import time")


def load_proxy(monkeypatch, state: Path):
    monkeypatch.setenv("SYNTHETIC_STATE", str(state))
    monkeypatch.setattr(http.server, "ThreadingHTTPServer", _NoServer)
    spec = importlib.util.spec_from_file_location("synthetic_proxy_under_test", PROXY_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    module = load_proxy(monkeypatch, tmp_path)
    module.write_material()
    names = ("ca.pem", "ca-key.pem", "signing-key.pem")
    first = {name: (tmp_path / name).read_bytes() for name in names}
    first_jwk = module.JWK

    module.write_material()

    for name in names:
        assert (tmp_path / name).read_bytes() == first[name], name
    assert module.JWK == first_jwk
    assert file_mode(tmp_path / "ca.pem") == 0o644
    for name in ("ca.pem", "ca-key.pem", "signing-key.pem", "server.pem", "server-key.pem"):
        assert (tmp_path / name).is_file()


def test_assertion_fields_expiry_and_signature(monkeypatch, tmp_path):
    module = load_proxy(monkeypatch, tmp_path)
    _ca_pem, jwk = module.write_material()

    payload = verify_assertion(module.mint_assertion(now=FROZEN_NOW), {"keys": [jwk]})

    assert payload["iss"] == "https://synthetic-team.cloudflareaccess.com"
    assert payload["aud"] == "raphael-server2-browser"
    assert payload["sub"] == "raphael-server2-owner"
    assert payload["type"] == "app"
    assert payload["email"] == "owner@server2.invalid"
    assert payload["iat"] == FROZEN_NOW
    assert payload["nbf"] == FROZEN_NOW - 30
    assert payload["exp"] == FROZEN_NOW + 600
    assert set(payload) - {"iss", "aud", "sub", "type", "email", "iat", "nbf", "exp"} <= {"jti"}


def test_two_requests_get_separately_minted_assertions(monkeypatch, tmp_path):
    module = load_proxy(monkeypatch, tmp_path)
    module.write_material()
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
    module = load_proxy(monkeypatch, tmp_path)
    old_umask = os.umask(0o000)
    try:
        module.write_material()
        for name in ("ca-key.pem", "signing-key.pem", "server-key.pem"):
            assert file_mode(tmp_path / name) & 0o077 == 0, name
        # A key left loose by something else is tightened on the next start.
        (tmp_path / "signing-key.pem").chmod(0o644)
        (tmp_path / "ca-key.pem").chmod(0o664)
        module.write_material()
        for name in ("ca-key.pem", "signing-key.pem", "server-key.pem"):
            assert file_mode(tmp_path / name) == 0o600, name
    finally:
        os.umask(old_umask)


def test_server_certificate_is_one_year_and_reissued_near_expiry(monkeypatch, tmp_path):
    module = load_proxy(monkeypatch, tmp_path)
    module.write_material()
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
