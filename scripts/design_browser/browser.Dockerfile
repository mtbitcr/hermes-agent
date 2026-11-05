# Look-only browser for the design roles, built and run by compose.yaml on the execution server (runbook:
# README.md in this folder). Same digest-pinned Playwright image as scripts/journeys/images/browser.Dockerfile;
# watchdog.sh is the main process.
FROM mcr.microsoft.com/playwright:v1.63.0-noble@sha256:eff16c30e6f3f4af0a03fa4b706120d5e9b0891c344a27d64559aff5900a4a27

# socat forwards the debugging port inside the container: headless Chromium listens on 127.0.0.1 only (README.md).
RUN apt-get update \
    && apt-get install -y --no-install-recommends socat \
    && rm -rf /var/lib/apt/lists/*

COPY watchdog.sh /usr/local/bin/watchdog.sh

HEALTHCHECK CMD ["curl", "-fsS", "--max-time", "5", "-o", "/dev/null", "http://127.0.0.1:9222/json/version"]

CMD ["/usr/local/bin/watchdog.sh"]
