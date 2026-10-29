#!/usr/bin/env bash
# One run of the scripted owner journeys (card P-3), inside the unit that start.sh starts, on Server 1.
# Steps, in this order, each at most once; the first that fails ends the run (no retry, no later step):
#   refresh    refresh_stack.sh of this folder: the stack is at the released versions (it reads
#              JOURNEY_RUNNER_SSH and JOURNEY_S2_SCRIPTS_DIR)
#   isolation  isolation_check.sh server2 on Server 2 through the runner identity, then server1 here
# then, for journeys 3, 5, 2, 1 and 4 in that order, each on a fresh stack:
#   time       refuse to start the journey when the minutes since the run started plus its fixed reserve
#              (reset 4, seed 1 for 3, 5 and 2 only, owner app 2, the journey's budget plus 1: 3 is 2, 5 is 6,
#              2 is 4, 1 is 21, 4 is 16) are more than 118
#   reset      reset_stack.sh on Server 2 through the runner identity
#   seed       seed_stack.sh <n> on Server 2 (journeys 3, 5 and 2 only)
#   proxy      the sign-in proxy answers its key list (the one answer that needs no owner app)
#   owner-app  the owner app shows JOURNEY_PAGE_TEXT, polled for 120 seconds at most
#   journeys   docker exec in the browser container on Server 2, through the runner identity:
#              sh tools/page-check/journey-run.sh <n> with only JOURNEY_BASE_URL and JOURNEY_PRODUCTION_ORIGIN passed in
# The proxy and owner-app answers are fetched with curl inside the browser container through the runner identity
# (the stack publishes no port) and checked here on Server 1. The owner app's root answers a redirect to
# /projects, so its check follows at most 3 redirects and looks for the page text on the page it lands on;
# the proxy check follows none.
# Once the run lock is held, a "started" result with its start time is stored; the terminal result
# replaces it (one file, RUN_DIR/result, replaced atomically); a failure in a journey's step also records
# journey=<n>. Then one owner line. A result or an owner
# line that cannot be stored or sent is its own failed step (result, notify), never retried; "passed" is
# only ever sent after "passed" was stored. Exit 0 only for a pass that was stored and told.
# Output of every step is dropped: the result holds a fixed reason, never a value or the page's text.
#
# Settings, set by the operator in the unit's environment (start.sh passes them on) unless noted:
#   HERMES_HOME                Server 1 platform home: the run folder is HERMES_HOME/journeys (lock and result),
#                              and the owner line is sent from it. Default ~/.hermes
#   JOURNEY_RUNNER_SSH         ssh destination of the runner identity on Server 2 (required)
#   JOURNEY_PAGE_TEXT          a phrase the owner app's page shows when it is up (required)
#   JOURNEY_RUNNER_ENV         Server 1 file with exactly two names, JOURNEY_BASE_URL (the owner app's address
#                              through the proxy) and JOURNEY_PRODUCTION_ORIGIN (the address the runner must
#                              never reach), as NAME=value lines; any other name refuses the run.
#                              Default ~/.hermes/journeys/runner.env; the operator creates it.
#   JOURNEY_S2_SCRIPTS_DIR     Server 2 directory holding the platform's scripts/journeys of the released commit;
#                              the operator installs them. Default /home/raphael-runner/raphael-journeys
#   JOURNEY_BROWSER_CONTAINER  the browser container on Server 2. Default raphael-journeys-browser
#   JOURNEY_BROWSER_DIR        directory in that container holding the owner app's tools/page-check of the
#                              released owner app commit; the operator installs it. Default /opt/raphael-journeys
# The operator also installs the proxy's certificate authority into the browser container's trust store (the
# browser's TLS checks stay on; this card does nothing about it), and sets JOURNEY_STACK_HOST and
# JOURNEY_MODEL_TOKEN_FILE in the runner identity's own environment on Server 2 for the two Server 2 scripts.
set -uo pipefail
umask 077
export LC_ALL=C TZ=UTC

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUN_DIR=${HERMES_HOME:-$HOME/.hermes}/journeys
RESULT=$RUN_DIR/result
PROXY_SOURCE=$HERE/synthetic_proxy.py
RUNNER_ENV=${JOURNEY_RUNNER_ENV:-$HOME/.hermes/journeys/runner.env}
S2_DIR=${JOURNEY_S2_SCRIPTS_DIR:-/home/raphael-runner/raphael-journeys}
CONTAINER=${JOURNEY_BROWSER_CONTAINER:-raphael-journeys-browser}
BROWSER_DIR=${JOURNEY_BROWSER_DIR:-/opt/raphael-journeys}
WAIT_SECONDS=120
POLL_SECONDS=2
LOOP_ORDER=(3 5 2 1 4)
LIMIT_MINUTES=118
RESET_MINUTES=4
SEED_MINUTES=1
APP_MINUTES=2
HTTPS_RE='^https://[A-Za-z0-9]([A-Za-z0-9.-]*[A-Za-z0-9])?(:[0-9]{1,5})?/?$'
SSH_RE='^[A-Za-z0-9_][A-Za-z0-9_.@-]*$'
DIR_RE='^/[A-Za-z0-9_./-]+$'
NAME_RE='^[A-Za-z0-9][A-Za-z0-9_.-]*$'

reason_for() {
  case $1 in
    config) echo "a required setting is missing or is not plain" ;;
    refresh) echo "the stack could not be brought to the released versions" ;;
    isolation) echo "the isolation check failed" ;;
    time) echo "the run's time ran out before the next journey" ;;
    reset) echo "the stack reset failed" ;;
    seed) echo "the seed failed" ;;
    proxy) echo "the sign-in proxy did not answer" ;;
    owner-app) echo "the owner app did not show its page text in time" ;;
    journeys) echo "the journey runner failed" ;;
    result) echo "the result file could not be stored" ;;
    notify) echo "the owner could not be told" ;;
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

JOURNEY=""
now() { printf '%(%Y-%m-%dT%H:%M:%SZ)T' -1; }
STARTED=$(now)
START_EPOCH=$(date +%s)

# store_result OUTCOME STEP REASON [EARLIER_STEP]: the whole result file, written beside the old one and
# moved over it, so that a reader sees the old file or the new one, never half of either.
store_result() {
  local outcome=$1 step=$2 reason=$3 earlier=${4-}
  {
    printf 'result=%s\nstep=%s\nreason=%s\nstarted=%s\n' "$outcome" "$step" "$reason" "$STARTED"
    if [[ -n $JOURNEY ]]; then printf 'journey=%s\n' "$JOURNEY"; fi
    if [[ $outcome != started ]]; then printf 'finished=%s\n' "$(now)"; fi
    if [[ -n $earlier ]]; then printf 'earlier_step=%s\n' "$earlier"; fi
  } > "$RESULT.new" 2> /dev/null && mv -f "$RESULT.new" "$RESULT" 2> /dev/null && return 0
  rm -f "$RESULT.new"
  return 1
}

tell() {
  printf '%s' "$1" | HERMES_SUPERVISED_CHILD=1 hermes send --to telegram --file - > /dev/null 2>&1
}

# finish OUTCOME STEP: the terminal result, then the one owner line; the exit status follows what happened.
finish() {
  local outcome=$1 step=$2 reason line stored=1 earlier=""
  if [[ $outcome == passed ]]; then
    reason="all steps passed"
    line="Journeys passed."
  else
    reason=$(reason_for "$step")
    line="Journeys failed at step $step${JOURNEY:+ for journey $JOURNEY}: $reason."
  fi
  if ! store_result "$outcome" "$step" "$reason"; then
    echo "run_once: the result file could not be stored" >&2
    stored=0
    if [[ $outcome == passed ]]; then
      outcome=failed
      step=result
      line="Journeys failed at step result: $(reason_for result)."
    elif [[ $step != result ]]; then
      line="$line The result file could not be stored."
    fi
  fi
  if ! tell "$line"; then
    echo "run_once: the owner was not told" >&2
    if ((stored)); then
      if [[ $outcome == failed ]]; then earlier=$step; fi
      # A result that cannot be amended must not stay "passed": it is taken away instead.
      store_result failed notify "$(reason_for notify)" "$earlier" || {
        rm -f "$RESULT" 2> /dev/null
        echo "run_once: the result file could not be amended" >&2
      }
    fi
    exit 1
  fi
  if [[ $outcome == passed && $stored -eq 1 ]]; then exit 0; fi
  exit 1
}

# The settings file holds the two names and no other: NAME=value lines, blank lines and # comments.
BASE_URL="" ORIGIN=""
read_runner_env() {
  local line name value
  [[ -f $RUNNER_ENV && -r $RUNNER_ENV ]] || return 1
  while IFS= read -r line || [[ -n $line ]]; do
    line=${line#"${line%%[![:space:]]*}"}
    line=${line%"${line##*[![:space:]]}"}
    [[ -z $line || $line == "#"* ]] && continue
    [[ $line =~ ^([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] || return 1
    name=${BASH_REMATCH[1]} value=${BASH_REMATCH[2]}
    [[ $value =~ $HTTPS_RE ]] || return 1
    case $name in
      JOURNEY_BASE_URL) [[ -z $BASE_URL ]] || return 1; BASE_URL=$value ;;
      JOURNEY_PRODUCTION_ORIGIN) [[ -z $ORIGIN ]] || return 1; ORIGIN=$value ;;
      *) return 1 ;;
    esac
  done < "$RUNNER_ENV"
  [[ -n $BASE_URL && -n $ORIGIN ]]
}

# The run is on record as started, with its time, before anything else happens.
store_result started none "the run has started and has not ended" || finish failed result

# No step starts before the settings that a later step needs are known to be plain: a reset must not
# be followed by a step that cannot run.
{
  [[ -n ${JOURNEY_PAGE_TEXT-} ]] && [[ ${JOURNEY_RUNNER_SSH-} =~ $SSH_RE ]] &&
    [[ $S2_DIR =~ $DIR_RE ]] && [[ $BROWSER_DIR =~ $DIR_RE ]] && [[ $CONTAINER =~ $NAME_RE ]] &&
    read_runner_env
} || finish failed config

# remote COMMAND: one command on Server 2 through the runner identity; its output is dropped.
remote() { ssh "$JOURNEY_RUNNER_SSH" "$1" > /dev/null 2>&1 < /dev/null; }
# remote_answer COMMAND: the same, with the command's standard output kept (its errors are dropped).
remote_answer() { ssh "$JOURNEY_RUNNER_SSH" "$1" 2> /dev/null < /dev/null; }

JOURNEY_RUNNER_SSH=$JOURNEY_RUNNER_SSH JOURNEY_S2_SCRIPTS_DIR=$S2_DIR \
  "$HERE/refresh_stack.sh" > /dev/null 2>&1 < /dev/null || finish failed refresh

remote "cd '$S2_DIR' && JOURNEY_PRODUCTION_ORIGIN='$ORIGIN' ./isolation_check.sh server2" || finish failed isolation
"$HERE/isolation_check.sh" server1 > /dev/null 2>&1 < /dev/null || finish failed isolation

# What a healthy sign-in proxy answers: its key list, with the key id that its source names.
proxy_answers() {
  local line issuer="" kid="" body
  while IFS= read -r line; do
    if [[ $line =~ ^ISSUER\ =\ \"([^\"]*)\"$ ]]; then issuer=${BASH_REMATCH[1]}; fi
    if [[ $line =~ ^KID\ =\ \"([^\"]*)\"$ ]]; then kid=${BASH_REMATCH[1]}; fi
  done < "$PROXY_SOURCE" || return 1
  [[ $issuer =~ $HTTPS_RE && -n $kid ]] || return 1
  body=$(remote_answer "docker exec '$CONTAINER' curl --disable --silent --fail --max-time 20 '${issuer%/}/cdn-cgi/access/certs'") || return 1
  [[ $body == *"\"kid\":\"$kid\""* ]]
}

# The page text is checked, not only the status: a broken stack answers 200 with a failure banner.
owner_app_shows_its_text() {
  local page
  page=$(remote_answer "docker exec '$CONTAINER' curl --disable --silent --fail --location --max-redirs 3 --max-time 10 '$BASE_URL'") || return 1
  [[ $page == *"$JOURNEY_PAGE_TEXT"* ]]
}

# reserve_minutes N: what journey N needs from its start to its end, the sum of its fixed step reserves.
reserve_minutes() {
  local total=$((RESET_MINUTES + APP_MINUTES))
  case $1 in 3 | 5 | 2) total=$((total + SEED_MINUTES)) ;; esac
  case $1 in
    3) echo $((total + 2)) ;;
    5) echo $((total + 6)) ;;
    2) echo $((total + 4)) ;;
    1) echo $((total + 21)) ;;
    4) echo $((total + 16)) ;;
  esac
}

there_is_time_for() {
  local at
  at=$(date +%s)
  [[ $at =~ ^[0-9]+$ && $START_EPOCH =~ ^[0-9]+$ ]] || return 1
  ((at - START_EPOCH + $(reserve_minutes "$1") * 60 <= LIMIT_MINUTES * 60))
}

for n in "${LOOP_ORDER[@]}"; do
  JOURNEY=$n
  there_is_time_for "$n" || finish failed time

  remote "cd '$S2_DIR' && ./reset_stack.sh" || finish failed reset

  case $n in
    3 | 5 | 2) remote "cd '$S2_DIR' && ./seed_stack.sh $n" || finish failed seed ;;
  esac

  proxy_answers || finish failed proxy

  waited=0
  until owner_app_shows_its_text; do
    if ((waited >= WAIT_SECONDS)); then finish failed owner-app; fi
    sleep "$POLL_SECONDS"
    waited=$((waited + POLL_SECONDS))
  done

  # docker takes the two values from its own environment: only the two names are passed in, no value.
  remote "JOURNEY_BASE_URL='$BASE_URL' JOURNEY_PRODUCTION_ORIGIN='$ORIGIN' docker exec -w '$BROWSER_DIR' -e JOURNEY_BASE_URL -e JOURNEY_PRODUCTION_ORIGIN '$CONTAINER' sh tools/page-check/journey-run.sh $n" ||
    finish failed journeys
done

JOURNEY=""
finish passed journeys
