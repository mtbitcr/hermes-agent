# Sign-in proxy of the journeys stack. Build folder: synthetic_proxy.py only.
# The digest below is a stand-in; set the resolved digest of python:3.12-slim before the first build.
FROM python:3.12-slim@sha256:0000000000000000000000000000000000000000000000000000000000000000

RUN pip install --no-cache-dir cryptography==50.0.0

# Docker copies this empty, user-owned folder into the proxy volume on first use, so the proxy can write its keys.
RUN groupadd --system --gid 10001 proxy \
    && useradd --system --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin proxy \
    && mkdir /state \
    && chown proxy:proxy /state

WORKDIR /app
COPY synthetic_proxy.py /app/synthetic_proxy.py

USER proxy
CMD ["python3", "synthetic_proxy.py"]
