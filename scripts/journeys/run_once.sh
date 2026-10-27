#!/usr/bin/env bash
# One run of the scripted owner journeys (card P-3), inside the unit that start.sh starts.
# Steps, in this order, each at most once; the first that fails ends the run:
#   isolation  isolation_check.sh server2, then server1
#   reset      reset_stack.sh
#   proxy      the sign-in proxy answers its key list (the one answer that needs no owner app)
#   owner-app  the owner app shows JOURNEY_PAGE_TEXT, polled for 120 seconds at most
#   journeys   journey-run.sh on Server 2 over ssh, through the runner identity (JOURNEY_RUNNER_SSH)
# Then one result file (RUN_DIR/result) and one owner line, for a pass and a fail alike. Exit 0 only for
# a pass that was recorded and told. Nothing is retried.
# Environment: JOURNEY_BASE_URL (the owner app address through the proxy), JOURNEY_PAGE_TEXT (a phrase
# the owner app's page shows when it is up), JOURNEY_RUNNER_SSH (the ssh destination of the runner
# identity); JOURNEY_STACK_HOST, JOURNEY_PRODUCTION_ORIGIN and JOURNEY_MODEL_TOKEN_FILE are for the
# two scripts. Output of every step is dropped: the result holds a fixed reason, never a value or the
# page's text.
set -uo pipefail
umask 077
export LC_ALL=C TZ=UTC

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUN_DIR=${HERMES_HOME:-$HOME/.hermes}/journeys
RESULT=$RUN_DIR/result
PROXY_SOURCE=$HERE/synthetic_proxy.py
WAIT_SECONDS=120
POLL_SECONDS=2
HTTPS_RE='^https://[A-Za-z0-9]([A-Za-z0-9.-]*[A-Za-z0-9])?(:[0-9]{1,5})?/?$'
SSH_RE='^[A-Za-z0-9_][A-Za-z0-9_.@-]*$'

reason_for() {
  case $1 in
    config) echo "a required setting is missing or is not plain" ;;
    isolation) echo "the isolation check failed" ;;
    reset) echo "the stack reset failed" ;;
    proxy) echo "the sign-in proxy did not answer" ;;
    owner-app) echo "the owner app did not show its page text in time" ;;
    journeys) echo "the journey runner failed" ;;
  esac
}

# The one run at a time: the lock is held to the end of this process, and goes with it.
mkdir -p "$RUN_DIR" && exec 9>> "$RUN_DIR/run.lock" || {
  echo "run_once: the folder of the run lock cannot be used" >&2
  exit 1
}
flock -n 9 || {
  echo "run_once: a journey run is already at work; nothing was run" >&2
  exit 1
}

# finish OUTCOME STEP: the result file, then the owner line; the exit status follows the outcome.
finish() {
  local outcome=$1 step=$2 reason line status=0
  if [[ $outcome == passed ]]; then
    reason="all steps passed"
    line="Journeys passed."
  else
    reason=$(reason_for "$step")
    line="Journeys failed at step $step: $reason."
  fi
  if printf 'result=%s\nstep=%s\nreason=%s\nfinished=%(%Y-%m-%dT%H:%M:%SZ)T\n' \
    "$outcome" "$step" "$reason" -1 > "$RESULT.new" && mv -f "$RESULT.new" "$RESULT"; then
    :
  else
    rm -f "$RESULT.new"
    echo "run_once: the result file could not be written" >&2
    status=1
  fi
  if ! printf '%s' "$line" | HERMES_SUPERVISED_CHILD=1 hermes send --to telegram --file - > /dev/null 2>&1; then
    echo "run_once: the owner was not told" >&2
    status=1
  fi
  if [[ $outcome == passed && $status -eq 0 ]]; then exit 0; fi
  exit 1
}

# No step starts before the settings that a later step needs are known to be plain: a reset must not
# be followed by a step that cannot run.
{ [[ ${JOURNEY_BASE_URL-} =~ $HTTPS_RE ]] && [[ -n ${JOURNEY_PAGE_TEXT-} ]] &&
  [[ ${JOURNEY_RUNNER_SSH-} =~ $SSH_RE ]]; } || finish failed config

"$HERE/isolation_check.sh" server2 > /dev/null 2>&1 < /dev/null || finish failed isolation
"$HERE/isolation_check.sh" server1 > /dev/null 2>&1 < /dev/null || finish failed isolation

"$HERE/reset_stack.sh" > /dev/null 2>&1 < /dev/null || finish failed reset

# What a healthy sign-in proxy answers: its key list, with the key id that its source names.
proxy_answers() {
  local line issuer="" kid="" body
  while IFS= read -r line; do
    if [[ $line =~ ^ISSUER\ =\ \"([^\"]*)\"$ ]]; then issuer=${BASH_REMATCH[1]}; fi
    if [[ $line =~ ^KID\ =\ \"([^\"]*)\"$ ]]; then kid=${BASH_REMATCH[1]}; fi
  done < "$PROXY_SOURCE" || return 1
  [[ $issuer =~ $HTTPS_RE && -n $kid ]] || return 1
  body=$(curl --disable --silent --fail --max-time 20 "${issuer%/}/cdn-cgi/access/certs" 2> /dev/null) || return 1
  [[ $body == *"\"kid\":\"$kid\""* ]]
}
proxy_answers || finish failed proxy

# The page text is checked, not only the status: a broken stack answers 200 with a failure banner.
owner_app_shows_its_text() {
  local page
  page=$(curl --disable --silent --fail --max-time 10 "$JOURNEY_BASE_URL" 2> /dev/null) || return 1
  [[ $page == *"$JOURNEY_PAGE_TEXT"* ]]
}
waited=0
until owner_app_shows_its_text; do
  if ((waited >= WAIT_SECONDS)); then finish failed owner-app; fi
  sleep "$POLL_SECONDS"
  waited=$((waited + POLL_SECONDS))
done

ssh "$JOURNEY_RUNNER_SSH" journey-run.sh > /dev/null 2>&1 < /dev/null || finish failed journeys

finish passed journeys
