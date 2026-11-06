#!/usr/bin/env bash
# Main process of the design browser container (browser.Dockerfile; runbook: README.md in this folder).
#
# It starts headless Chromium with remote debugging on 127.0.0.1:9222, forwards that port to the
# container's own address and checks both. It exits when Chromium or the forward dies, when Chromium
# fails three /json/version checks in a row, and after 6 hours without a client. It exits instead of
# restarting Chromium itself because rootless Docker restarts a container only on exit: restart:
# unless-stopped (compose.yaml) then starts the container again with a new Chromium and an empty profile.
#
# Timings: a dead Chromium is seen by the next process check, within 15 s. A frozen one is still
# running but fails the /json/version check that is due within 60 s; a failed check is repeated every
# 15 s, so the third failure, and the exit, comes at most 60 + 5 + 2 * (15 + 5) = 105 s after the freeze.
# The image's HEALTHCHECK first runs 30 s after the restart, so either is healthy again within 3 minutes.
set -euo pipefail

port=9222
shallow_every=15           # seconds between process checks
deep_every=60              # seconds between /json/version checks while they pass
deep_timeout=5             # seconds one /json/version check may take
deep_failures_max=3        # failed /json/version checks in a row before the exit
idle_max=$((6 * 60 * 60))  # seconds without a client before Chromium is recycled

log() { echo "watchdog: $*"; }

chrome=("$PLAYWRIGHT_BROWSERS_PATH"/chromium-*/chrome-linux*/chrome)
switches=(
  --headless
  --remote-debugging-port="$port"
  --user-data-dir=/profile  # a tmpfs (compose.yaml), empty at every start
  # The navigation guard becomes active only when Browser.getBrowserCommandLine shows
  # --block-new-web-contents, and with that switch no page can open a second window. This is the
  # owner's rule for the navigation guard. Chromium answers that call only with --enable-automation.
  --block-new-web-contents
  --enable-automation
  # cap_drop ALL leaves Chromium's sandbox no way to start: without this it aborts, "No usable sandbox!".
  --no-sandbox
)
# The root is read-only: with an unwritable HOME, Chromium aborts (its crash handler keeps a database there).
HOME=/tmp "${chrome[0]}" "${switches[@]}" &
chromium_pid=$!
log "started ${chrome[0]}, pid $chromium_pid"

# Headless Chromium listens on 127.0.0.1 only, whatever --remote-debugging-address says (README.md), but
# the published port arrives on the container's own address. socat forwards that address's port to
# Chromium (0.0.0.0 cannot be bound: Chromium already holds 127.0.0.1 on the same port).
address=$(hostname -I | awk '{ for (i = 1; i <= NF; i++) if ($i ~ /^[0-9.]+$/) { print $i; exit } }')
socat "TCP4-LISTEN:$port,bind=$address,fork" "TCP4:127.0.0.1:$port" &
forward_pid=$!
log "forwarding $address:$port to 127.0.0.1:$port, pid $forward_pid"

# A client is attached while socat forwards one of its connections: socat serves each in a child process.
client_attached() {
  local stat line
  for stat in /proc/[0-9]*/stat; do
    read -r line 2>/dev/null <"$stat" || continue
    line=${line##*") "}  # the fields after "pid (name) ": state, parent pid, ...
    [[ ${line#* } == "$forward_pid "* ]] && return 0
  done
  return 1
}

deep_failures=0
deep_due=0
idle_since=$SECONDS
while true; do
  sleep "$shallow_every"
  kill -0 "$chromium_pid" 2>/dev/null || { log "Chromium died"; exit 1; }
  kill -0 "$forward_pid" 2>/dev/null || { log "the port forward died"; exit 1; }

  if (( deep_failures > 0 || SECONDS >= deep_due )); then
    if curl -fsS --max-time "$deep_timeout" -o /dev/null "http://127.0.0.1:$port/json/version"; then
      deep_failures=0
    else
      deep_failures=$((deep_failures + 1))
      log "/json/version check failed ($deep_failures of $deep_failures_max)"
      if (( deep_failures >= deep_failures_max )); then exit 1; fi
    fi
    deep_due=$((SECONDS + deep_every))
  fi

  if client_attached; then
    idle_since=$SECONDS
  elif (( SECONDS - idle_since >= idle_max )); then
    log "no client for 6 hours: exiting so that Docker starts a new Chromium"
    exit 0
  fi
done
