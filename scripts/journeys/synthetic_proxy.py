from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
import ssl
import time

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.x509.oid import NameOID


STATE = Path(os.environ.get("SYNTHETIC_STATE", "/state"))
ISSUER = "https://synthetic-team.cloudflareaccess.com"
AUDIENCE = "raphael-server2-browser"
SUBJECT = "raphael-server2-owner"
KID = "raphael-server2-key"
WORKSPACE_HOST = "workspace.synthetic.test"
CA_VALIDITY = timedelta(days=3650)
SERVER_VALIDITY = timedelta(days=365)
SERVER_RENEW_WITHIN = timedelta(days=30)
ASSERTION_LIFETIME = 600

CA_PEM: bytes = b""
JWK: dict = {}
SIGNING_KEY: rsa.RSAPrivateKey | None = None


def b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _write_file(path: Path, data: bytes, mode: int) -> None:
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "wb") as handle:
            fd = -1
            handle.write(data)
    finally:
        if fd != -1:
            os.close(fd)
    os.replace(tmp, path)


def _key_pem(key: rsa.RSAPrivateKey) -> bytes:
    return key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )


def _load_private_key(path: Path) -> rsa.RSAPrivateKey:
    path.chmod(0o600)
    return serialization.load_pem_private_key(path.read_bytes(), password=None)


def _load_or_create_key(path: Path) -> rsa.RSAPrivateKey:
    if path.exists():
        return _load_private_key(path)
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    _write_file(path, _key_pem(key), 0o600)
    return key


def _load_or_create_ca(state: Path, now: datetime) -> tuple[x509.Certificate, rsa.RSAPrivateKey]:
    cert_path, key_path = state / "ca.pem", state / "ca-key.pem"
    if cert_path.exists() and key_path.exists():
        ca_key = _load_private_key(key_path)
        ca_cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
        cert_path.chmod(0o644)
        return ca_cert, ca_key
    ca_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Raphael Server 2 Test CA")])
    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(ca_name)
        .issuer_name(ca_name)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + CA_VALIDITY)
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(ca_key, hashes.SHA256())
    )
    _write_file(key_path, _key_pem(ca_key), 0o600)
    _write_file(cert_path, ca_cert.public_bytes(serialization.Encoding.PEM), 0o644)
    return ca_cert, ca_key


def _server_material_is_current(
    state: Path, ca_cert: x509.Certificate, now: datetime
) -> bool:
    try:
        server_cert = x509.load_pem_x509_certificate((state / "server.pem").read_bytes())
        server_key = _load_private_key(state / "server-key.pem")
        server_cert.verify_directly_issued_by(ca_cert)
    except Exception:
        return False
    return (
        server_cert.not_valid_after_utc - now > SERVER_RENEW_WITHIN
        and server_cert.public_key().public_numbers() == server_key.public_key().public_numbers()
    )


def _issue_server_material(
    state: Path, ca_cert: x509.Certificate, ca_key: rsa.RSAPrivateKey, now: datetime
) -> None:
    server_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    server_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, WORKSPACE_HOST)])
    server_cert = (
        x509.CertificateBuilder()
        .subject_name(server_name)
        .issuer_name(ca_cert.subject)
        .public_key(server_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + SERVER_VALIDITY)
        .add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName(WORKSPACE_HOST),
                x509.DNSName("synthetic-team.cloudflareaccess.com"),
            ]),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256())
    )
    _write_file(state / "server-key.pem", _key_pem(server_key), 0o600)
    _write_file(state / "server.pem", server_cert.public_bytes(serialization.Encoding.PEM), 0o644)


def write_material(state: Path | None = None) -> tuple[bytes, dict]:
    global CA_PEM, JWK, SIGNING_KEY
    state = STATE if state is None else Path(state)
    state.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    ca_cert, ca_key = _load_or_create_ca(state, now)
    if not _server_material_is_current(state, ca_cert, now):
        _issue_server_material(state, ca_cert, ca_key, now)
    SIGNING_KEY = _load_or_create_key(state / "signing-key.pem")
    numbers = SIGNING_KEY.public_key().public_numbers()
    JWK = {
        "kty": "RSA",
        "kid": KID,
        "alg": "RS256",
        "use": "sig",
        "n": b64url(numbers.n.to_bytes((numbers.n.bit_length() + 7) // 8, "big")),
        "e": b64url(numbers.e.to_bytes((numbers.e.bit_length() + 7) // 8, "big")),
    }
    CA_PEM = ca_cert.public_bytes(serialization.Encoding.PEM)
    return CA_PEM, JWK


def mint_assertion(now: float | None = None) -> str:
    header = b64url(json.dumps(
        {"alg": "RS256", "kid": KID, "typ": "JWT"},
        separators=(",", ":"),
    ).encode())
    issued = int(time.time() if now is None else now)
    payload = b64url(json.dumps({
        "iss": ISSUER,
        "aud": AUDIENCE,
        "sub": SUBJECT,
        "type": "app",
        "email": "owner@server2.invalid",
        "iat": issued,
        "nbf": issued - 30,
        "exp": issued + ASSERTION_LIFETIME,
        "jti": secrets.token_hex(16),
    }, separators=(",", ":")).encode())
    signing_input = f"{header}.{payload}".encode()
    signature = SIGNING_KEY.sign(signing_input, padding.PKCS1v15(), hashes.SHA256())
    return f"{header}.{payload}.{b64url(signature)}"


HOP_HEADERS = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "content-length",
}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, _format: str, *_args) -> None:
        return

    def _handle(self) -> None:
        hostname = (self.headers.get("host") or "").split(":", 1)[0].lower()
        if (
            hostname == "synthetic-team.cloudflareaccess.com"
            and self.path == "/cdn-cgi/access/certs"
        ):
            body = json.dumps({"keys": [JWK]}, separators=(",", ":")).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("cache-control", "no-store")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if hostname != WORKSPACE_HOST:
            self.send_error(404)
            return

        length = int(self.headers.get("content-length") or "0")
        request_body = self.rfile.read(length) if length else None
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in HOP_HEADERS
        }
        headers["host"] = WORKSPACE_HOST
        headers["x-forwarded-proto"] = "https"
        headers["cf-access-jwt-assertion"] = mint_assertion()
        upstream = http.client.HTTPConnection("workspace", 3000, timeout=15)
        try:
            upstream.request(self.command, self.path, request_body, headers)
            response = upstream.getresponse()
            body = response.read()
            self.send_response(response.status)
            for key, value in response.getheaders():
                if key.lower() not in HOP_HEADERS:
                    self.send_header(key, value)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)
        finally:
            upstream.close()

    do_GET = _handle
    do_HEAD = _handle
    do_POST = _handle
    do_PUT = _handle
    do_PATCH = _handle
    do_DELETE = _handle


def main() -> None:
    write_material()
    server = ThreadingHTTPServer(("0.0.0.0", 443), Handler)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(STATE / "server.pem", STATE / "server-key.pem")
    server.socket = context.wrap_socket(server.socket, server_side=True)
    print("READY", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
