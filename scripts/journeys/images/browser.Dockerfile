# Browser of the journeys stack. Build folder: ca.pem only (the proxy's public CA certificate).
# The page-check is not copied here. Playwright goes in /opt/raphael-journeys, so that node finds it from
# the page-check folder below it. The digest is the registry index digest read by the operator on 2026-10-28
# (metadata only, no image pulled); the platform's production Python is 3.11.15.
FROM mcr.microsoft.com/playwright:v1.63.0-noble@sha256:eff16c30e6f3f4af0a03fa4b706120d5e9b0891c344a27d64559aff5900a4a27

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libnss3-tools \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/raphael-journeys
RUN npm install --save-exact --no-audit --no-fund playwright@1.63.0

# System store (curl in the run's checks) and the NSS database of the default user (Chromium).
COPY ca.pem /usr/local/share/ca-certificates/raphael-journeys-proxy-ca.crt
RUN update-ca-certificates \
    && mkdir -p "$HOME/.pki/nssdb" \
    && certutil -d "sql:$HOME/.pki/nssdb" -N --empty-password \
    && certutil -d "sql:$HOME/.pki/nssdb" -A -t "C,," -n raphael-journeys-proxy-ca -i /usr/local/share/ca-certificates/raphael-journeys-proxy-ca.crt

CMD ["sleep", "2147483647"]
