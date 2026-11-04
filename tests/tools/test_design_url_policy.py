"""Address policy for the look-only design browser (plan v2 D1).

DNS is stubbed in every test: a name answers ``PUBLIC_IP`` unless the test maps it to other
addresses, so a refusal comes from the policy and never from a failed lookup.
"""

import ipaddress
import socket

import pytest

import tools.url_safety as url_safety
from hermes_cli.config_defaults import DEFAULT_CONFIG
from hermes_constants import get_hermes_home
from tools.design_url_policy import design_url_refusal
from tools.url_safety import is_safe_url, sensitive_query_param_name

PUBLIC_IP = "93.184.216.34"
PREVIEW = "raphael-first-project.pages.dev"
# The plan's hosts; the real allow-list is filled in later by another card.
PLAN_HOSTS = ["raphael.agentsbridge.net", PREVIEW, f"*.{PREVIEW}"]
PLAN_URLS = ["https://raphael.agentsbridge.net/", f"https://{PREVIEW}/", f"https://main.{PREVIEW}/"]
COUNTER_ENTRIES = [
    "static.cloudflareinsights.com",
    "cloudflareinsights.com",
    "*.cloudflareinsights.com",
    "*.static.cloudflareinsights.com",
]
HOMOGLYPH_HOST = "r\u0430phael.agentsbridge.net"  # Cyrillic small a
HOMOGLYPH_PREVIEW = "raphael-first-pr\u043eject.pages.dev"  # Cyrillic small o
SECRET = "s3cr3t-value"


def _punycode(host):
    return host.encode("idna").decode("ascii")


@pytest.fixture(autouse=True)
def dns(monkeypatch):
    """Stub ``socket.getaddrinfo``, which ``tools.url_safety`` calls for every lookup. An address
    answers itself, a name in the returned mapping answers its addresses, any other name answers
    ``PUBLIC_IP``."""
    answers = {}

    def getaddrinfo(host, *_args, **_kwargs):
        try:
            ipaddress.ip_address(host)
            ips = [host]
        except ValueError:
            ips = answers.get(host, [PUBLIC_IP])
        return [
            (socket.AF_INET6, socket.SOCK_STREAM, 6, "", (ip, 0, 0, 0)) if ":" in ip
            else (socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 0))
            for ip in ips
        ]

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    return answers


@pytest.fixture(autouse=True)
def _no_proxy_and_fresh_toggle(monkeypatch):
    """No proxy and no env override; ``tools.url_safety`` reads ``allow_private_urls`` afresh in
    each test, and monkeypatch puts its cached value back afterwards."""
    for var in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy",
                "HERMES_ALLOW_PRIVATE_URLS"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(url_safety, "_allow_private_resolved", False)
    monkeypatch.setattr(url_safety, "_cached_allow_private", False)


@pytest.fixture
def allow_private_urls(tmp_path):
    """``security.allow_private_urls: true`` in config.yaml, the file ``tools.url_safety`` reads.
    HERMES_HOME is the per-test home from tests/conftest.py, never the real ~/.hermes."""
    home = get_hermes_home()
    assert home.is_relative_to(tmp_path)
    (home / "config.yaml").write_text("security:\n  allow_private_urls: true\n", encoding="utf-8")


@pytest.mark.parametrize("policy", [
    pytest.param(PLAN_HOSTS, id="allow-list"),
    pytest.param({"design_browser": {"allowed_hosts": PLAN_HOSTS}}, id="config"),
])
@pytest.mark.parametrize("url", [
    "https://raphael.agentsbridge.net/",
    "https://raphael.agentsbridge.net/boards/design?page=2#top",
    "https://RAPHAEL.AgentsBridge.NET/",
    "https://raphael.agentsbridge.net:443/",
    f"https://{PREVIEW}/",
    f"https://main.{PREVIEW}/assets/app.js",
    f"https://b4-card-12.{PREVIEW}/",
])
def test_plan_hosts_are_allowed(url, policy):
    assert design_url_refusal(url, policy) is None


@pytest.mark.parametrize("url", [f"https://a.main.{PREVIEW}/", f"https://x.y.z.{PREVIEW}/"])
def test_preview_two_or_more_labels_down_is_denied(url):
    assert design_url_refusal(url, PLAN_HOSTS) is not None


@pytest.mark.parametrize("url", [
    "https://agentsbridge.net/",
    "https://www.raphael.agentsbridge.net/",
    "https://evilraphael.agentsbridge.net/",
    "https://raphael-agentsbridge.net/",
    "https://raphael.agentsbridge.net.evil.example/",
    "https://pages.dev/",
    "https://evil-raphael-first-project.pages.dev/",
    "https://raphael-first-project-evil.pages.dev/",
    f"https://{PREVIEW}.evil.example/",
    f"https://main.{PREVIEW}.evil.example/",
])
def test_look_alike_hosts_are_denied(url):
    assert design_url_refusal(url, PLAN_HOSTS) is not None


@pytest.mark.parametrize("url", [
    pytest.param(f"https://{HOMOGLYPH_HOST}/", id="homoglyph"),
    pytest.param(f"https://{_punycode(HOMOGLYPH_HOST)}/", id="homoglyph-punycode"),
    pytest.param(f"https://main.{HOMOGLYPH_PREVIEW}/", id="homoglyph-preview"),
    pytest.param(f"https://main.{_punycode(HOMOGLYPH_PREVIEW)}/", id="homoglyph-preview-punycode"),
    # Python's IDNA codec quietly maps each of these onto an allowed host; only a host that is
    # already canonical is compared, so they are refused.
    pytest.param("https://raphael-first-project\uff0epages.dev/", id="fullwidth-dot"),
    pytest.param("https://raphael\u3002agentsbridge.net/", id="ideographic-dot"),
    pytest.param("https://\uff52aphael.agentsbridge.net/", id="fullwidth-letter"),
    pytest.param("https://rapha\u200del.agentsbridge.net/", id="zero-width-joiner"),
    pytest.param("https://rapha\u00adel.agentsbridge.net/", id="soft-hyphen"),
    pytest.param(f"https://stra\u00dfe.{PREVIEW}/", id="sharp-s"),
    # Punycode that does not decode, or decodes to plain ASCII.
    pytest.param(f"https://xn--zz.{PREVIEW}/", id="broken-punycode"),
    pytest.param(f"https://xn--abc-.{PREVIEW}/", id="ascii-punycode"),
    # A fullwidth number sign becomes "#" under NFKC, which would end the host early.
    pytest.param("https://raphael.agentsbridge.net\uff03.evil.example/", id="nfkc-invalid"),
])
def test_idn_and_punycode_tricks_are_denied(url):
    assert design_url_refusal(url, PLAN_HOSTS) is not None


@pytest.mark.parametrize("url, hosts", [
    ("https://b\u00fccher.example/", ["xn--bcher-kva.example"]),
    ("https://B\u00dcCHER.example/", ["xn--bcher-kva.example"]),
    ("https://xn--bcher-kva.example/", ["b\u00fccher.example"]),
])
def test_hosts_are_compared_in_punycode_form(url, hosts):
    assert design_url_refusal(url, hosts) is None


@pytest.mark.parametrize("url", [
    "https://user@raphael.agentsbridge.net/",
    "https://user:hunter2@raphael.agentsbridge.net/",
    "https://:@raphael.agentsbridge.net/",
    "https://evil.example@raphael.agentsbridge.net/",
    "https://raphael.agentsbridge.net@evil.example/",
])
def test_user_info_is_denied(url):
    refusal = design_url_refusal(url, PLAN_HOSTS)
    assert refusal is not None
    assert "hunter2" not in refusal


@pytest.mark.parametrize("port", ["8443", "80", "444", "0", "99999", ""])
@pytest.mark.parametrize("host", ["raphael.agentsbridge.net", f"main.{PREVIEW}"])
def test_non_default_ports_are_denied(host, port):
    assert design_url_refusal(f"https://{host}:{port}/", PLAN_HOSTS) is not None


@pytest.mark.parametrize("url", [
    "http://raphael.agentsbridge.net/",
    "file:///etc/passwd",
    "file://raphael.agentsbridge.net/etc/passwd",
    "data:text/html,<h1>design</h1>",
    "blob:https://raphael.agentsbridge.net/6f1c0f3e-8d2a-4b8e-9f00-3a1c2b4d5e6f",
    "javascript:alert(document.domain)",
    "chrome://settings/",
    "view-source:https://raphael.agentsbridge.net/",
])
def test_schemes_other_than_https_are_denied(url):
    assert design_url_refusal(url, PLAN_HOSTS) is not None


@pytest.mark.parametrize("url", [
    pytest.param(" https://raphael.agentsbridge.net/", id="leading-space"),
    pytest.param("https://raphael.agentsbridge.net/\n", id="trailing-newline"),
    pytest.param("https://raph\tael.agentsbridge.net/", id="tab-in-host"),
    pytest.param("https://raphael.agentsbridge.net\x00/", id="nul"),
    # A browser reads backslashes as "/", so this would request /cdn-cgi/rum.
    pytest.param("https://raphael.agentsbridge.net\\cdn-cgi\\rum", id="backslash"),
    pytest.param("https://raphael%2eagentsbridge.net/", id="percent-encoded-host"),
    pytest.param("https://raphael.agentsbridge.net./", id="trailing-dot"),
    pytest.param("https:raphael.agentsbridge.net/", id="no-slashes"),
    pytest.param("https:///raphael.agentsbridge.net/", id="three-slashes"),
    pytest.param(f"https://[va.{PREVIEW}]/", id="bracketed-name"),
])
def test_urls_that_do_not_parse_cleanly_are_denied(url):
    assert design_url_refusal(url, PLAN_HOSTS) is not None


@pytest.mark.parametrize("url", [
    "https://static.cloudflareinsights.com/beacon.min.js",
    "https://cloudflareinsights.com/",
    "https://beacon.cloudflareinsights.com/",
    "https://a.static.cloudflareinsights.com/",
])
def test_counter_hosts_are_denied_even_when_listed(url):
    hosts = [*PLAN_HOSTS, *COUNTER_ENTRIES]
    assert design_url_refusal(PLAN_URLS[0], hosts) is None
    assert design_url_refusal(url, hosts) is not None


@pytest.mark.parametrize("path", [
    "/cdn-cgi/rum",
    "/cdn-cgi/rum?ts=1",
    "/cdn-cgi/rum/",
    "/cdn-cgi/rum;x",
    "/CDN-CGI/RUM",
    "//cdn-cgi//rum",
    "/assets/../cdn-cgi/rum",
    "/cdn-cgi/./rum",
    "/%63dn-cgi/rum",
    "/cdn-cgi%2Frum",
    "/cdn-cgi/%2572um",
])
@pytest.mark.parametrize("host", ["raphael.agentsbridge.net", PREVIEW, f"main.{PREVIEW}"])
def test_counter_path_is_denied_on_every_host(host, path):
    assert design_url_refusal(f"https://{host}/", PLAN_HOSTS) is None
    assert design_url_refusal(f"https://{host}{path}", PLAN_HOSTS) is not None


@pytest.mark.parametrize("ips", [
    ["10.0.0.8"],
    ["172.16.0.5"],
    ["192.168.1.10"],
    ["127.0.0.1"],
    ["::1"],
    ["fe80::1"],
    ["100.64.0.1"],
    ["::ffff:10.0.0.8"],
    [PUBLIC_IP, "10.0.0.8"],
], ids="+".join)
def test_private_answers_are_denied_with_allow_private_urls_on(allow_private_urls, dns, ips):
    dns["raphael.agentsbridge.net"] = ips
    url = "https://raphael.agentsbridge.net/"
    assert is_safe_url(url)  # the toggle is on, so url_safety alone lets this through
    assert design_url_refusal(url, PLAN_HOSTS) is not None


@pytest.mark.parametrize("ip", [
    "169.254.169.254", "169.254.170.2", "100.100.100.200", "fd00:ec2::254", "::ffff:169.254.169.254",
])
def test_metadata_answers_are_denied_with_allow_private_urls_on(allow_private_urls, dns, ip):
    dns["raphael.agentsbridge.net"] = [ip]
    assert design_url_refusal("https://raphael.agentsbridge.net/", PLAN_HOSTS) is not None


@pytest.mark.parametrize("target", [
    "127.0.0.1",
    "[::1]",
    "10.0.0.1",
    "192.168.0.10",
    "169.254.169.254",
    "[fd00:ec2::254]",
    "metadata.google.internal",
    "localhost",
])
def test_address_targets_are_denied_even_when_listed(allow_private_urls, dns, target):
    dns["localhost"] = ["127.0.0.1", "::1"]
    hosts = [*PLAN_HOSTS, target, target.strip("[]")]
    assert design_url_refusal(f"https://{target}/", hosts) is not None


def test_unresolvable_host_is_denied_even_through_a_proxy(monkeypatch):
    def no_dns(*_args, **_kwargs):
        raise socket.gaierror("Name or service not known")

    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.internal:3128")
    monkeypatch.setattr(socket, "getaddrinfo", no_dns)
    url = "https://raphael.agentsbridge.net/"
    assert is_safe_url(url)  # url_safety leaves resolution to the proxy
    assert design_url_refusal(url, PLAN_HOSTS) is not None


@pytest.mark.parametrize("query", [
    f"token={SECRET}",
    f"page=2&api_key={SECRET}",
    f"Access_Token={SECRET}",
    f"%74oken={SECRET}",
])
def test_secret_named_query_parameter_is_denied(query):
    url = f"https://raphael.agentsbridge.net/boards?{query}"
    assert sensitive_query_param_name(url)
    refusal = design_url_refusal(url, PLAN_HOSTS)
    assert refusal is not None
    assert SECRET not in refusal


@pytest.mark.parametrize("url", [
    f"https://raphael.agentsbridge.net/?token={SECRET}",
    f"http://raphael.agentsbridge.net/?token={SECRET}",
    f"https://raphael.agentsbridge.net:8443/?token={SECRET}",
    f"https://user@raphael.agentsbridge.net/?token={SECRET}",
    f"https://evil.example/?token={SECRET}",
    f"https://raphael.agentsbridge.net/cdn-cgi/rum?token={SECRET}",
    f"https://{HOMOGLYPH_HOST}/?token={SECRET}",
])
def test_refusals_never_echo_the_query_value(url):
    refusal = design_url_refusal(url, PLAN_HOSTS)
    assert refusal
    assert SECRET not in refusal


@pytest.mark.parametrize("builder_host", [f"main.{PREVIEW}", f"MAIN.{PREVIEW.upper()}"])
def test_builder_narrowing_allows_exactly_one_host(builder_host):
    assert design_url_refusal(
        f"https://main.{PREVIEW}/assets/app.js", PLAN_HOSTS, builder_host=builder_host) is None
    for url in [
        "https://raphael.agentsbridge.net/",
        f"https://{PREVIEW}/",
        f"https://other.{PREVIEW}/",
        f"https://a.main.{PREVIEW}/",
    ]:
        assert design_url_refusal(url, PLAN_HOSTS, builder_host=builder_host) is not None


@pytest.mark.parametrize("builder_host, hosts", [
    pytest.param("docs.example", PLAN_HOSTS, id="unlisted"),
    pytest.param(f"a.main.{PREVIEW}", PLAN_HOSTS, id="two-labels-down"),
    pytest.param(f"*.{PREVIEW}", PLAN_HOSTS, id="wildcard"),
    pytest.param("", PLAN_HOSTS, id="empty"),
    pytest.param(f"https://main.{PREVIEW}/", PLAN_HOSTS, id="url-not-host"),
    pytest.param(f"main.{PREVIEW}", [], id="empty-allow-list"),
    pytest.param(
        "static.cloudflareinsights.com", [*PLAN_HOSTS, "static.cloudflareinsights.com"], id="counter"),
])
def test_builder_narrowing_never_widens(builder_host, hosts):
    for url in [*PLAN_URLS, f"https://{builder_host}/"]:
        assert design_url_refusal(url, hosts, builder_host=builder_host) is not None


@pytest.mark.parametrize("policy", [
    pytest.param(None, id="none"),
    pytest.param({}, id="no-section"),
    pytest.param({"design_browser": None}, id="section-none"),
    pytest.param({"design_browser": "raphael.agentsbridge.net"}, id="section-not-mapping"),
    pytest.param({"design_browser": {}}, id="no-allowed-hosts"),
    pytest.param({"design_browser": {"allowed_hosts": None}}, id="allowed-hosts-none"),
    pytest.param({"design_browser": {"allowed_hosts": []}}, id="allowed-hosts-empty"),
    pytest.param({"design_browser": {"allowed_hosts": "raphael.agentsbridge.net"}},
                 id="allowed-hosts-string"),
    pytest.param({"design_browser": {"allowed_hosts": {"raphael.agentsbridge.net": True}}},
                 id="allowed-hosts-mapping"),
    pytest.param({"design_browser": {"allowed_hosts": [None, 7, ["raphael.agentsbridge.net"]]}},
                 id="no-string-entries"),
    pytest.param({"allowed_hosts": PLAN_HOSTS}, id="section-contents-without-section"),
    pytest.param([], id="empty-allow-list"),
    pytest.param("raphael.agentsbridge.net", id="string-allow-list"),
    pytest.param(DEFAULT_CONFIG, id="default-config"),
])
def test_empty_or_malformed_config_denies_everything(policy):
    for url in PLAN_URLS:
        assert design_url_refusal(url, policy) is not None


def test_design_browser_settings_default_to_empty():
    section = DEFAULT_CONFIG["design_browser"]
    assert section["cdp_url"] == ""
    assert section["allowed_hosts"] == []
    assert section["boards"] == []
