# Sign-in proxy of the journeys stack. Build folder: synthetic_proxy.py only.
# The digest is the registry index digest read by the operator on 2026-10-28 (metadata only, no image pulled); the platform's production Python is 3.11.15.
FROM python:3.11.15-slim-bookworm@sha256:d29f48a31a8b408ed19272ca1e7b10ebae13b240a27e862d3d4217c528e2e0c3

RUN pip install --no-cache-dir cryptography==50.0.0

# Docker copies this empty, user-owned folder into the proxy volume on first use, so the proxy can write its keys.
RUN groupadd --system --gid 10001 journeys-proxy \
    && useradd --system --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin journeys-proxy \
    && mkdir /state \
    && chown journeys-proxy:journeys-proxy /state

WORKDIR /app
COPY synthetic_proxy.py /app/synthetic_proxy.py

USER journeys-proxy
CMD ["python3", "synthetic_proxy.py"]
