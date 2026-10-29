#!/usr/bin/env bash
# Bring the journeys test stack on Server 2 to the released versions (card 3). Runs on Server 1, takes no argument, and
# exits 0 only when the stack is at the released versions.
#
# The released versions:
#   platform  the commit of the checkout that holds this script (git, found from this script's own folder). It must
#             equal code_sha in HERMES_HOME/gateway_state.json (read without jq or python); otherwise the run is refused
#             before any Server 2 step.
#   owner app the org.opencontainers.image.revision label and the image id of the production container named in
#             [production-container] of stack.env.template, from one docker inspect. On Server 1 this script runs only
#             that docker inspect and docker save of that image id; it changes nothing.
#
# Settings: JOURNEY_RUNNER_SSH (ssh destination of the runner identity on Server 2; required), JOURNEY_S2_SCRIPTS_DIR
# (Server 2 folder of scripts/journeys, called S2dir here; same rule and default as run_once.sh; a trailing slash is
# stripped; the root is refused), HERMES_HOME (default ~/.hermes). Container names come from stack.env.template next to
# this script. Every Server 2 step is one ssh "$JOURNEY_RUNNER_SSH" "<command>" with exactly two arguments; no command
# line carries a value.
#
# The box is 25 minutes from the start. Every git, docker and ssh call (both sides of a pipeline) runs as
# timeout <seconds left> <command> and is not started when no time is left. The clock is read before and after each call;
# a call that comes back at or after the box is late even when it succeeded. A status of 124 or 137, or a failure once the
# box is over, ends the run with reason box. Nothing on Server 2 is killed, stopped or removed for it.
#
# Reads on Server 2 (a failed or garbled read is never an absence, a difference or "current"):
#   marker  S2dir/.refresh-commit. Status 0 and exactly one line of 40 lowercase hex digits (the final newline may be
#           missing): this commit or another commit. Status 3, from a folder that can be searched and has no such entry
#           or from a missing folder whose parent can be searched: no marker. Everything else ends the run: marker-ask.
#   image   one docker image inspect of the exact id or reference, parsed here and never printed. Found: status 0 and one
#           line sha256: plus 64 lowercase hex (for an id, the id asked). Missing: status 1 and Docker's own not-found
#           line for exactly that name (after at most one empty line or "[]" line). Everything else ends the run:
#           image-ask. docker image ls decides no existence; step g uses it to list tags.
# A read may come after a transfer. A run that ends with exit 1 after a transfer is accepted: the next run repeats it.
#
# Steps, in this order:
#   0  first look, three read-only ssh calls, each only when the one before agreed: the marker holds this commit,
#      stack_up.sh versions shows the released versions, the page check folder is in the browser container. When all
#      three agree nothing is shipped, built or converged and only step g runs. Otherwise the stack is not current.
#   a  unless the marker holds this commit: scripts/journeys of the commit, as a git archive, is extracted into S2dir.new.
#      A second call then writes .refresh-commit into it and swaps it in with two renames (an old folder stays as
#      S2dir.previous; when the second rename fails the first is undone). Only that call writes the marker.
#   b  the owner app image: shipped (docker save into docker load, which leaves an id without a tag) when image inspect of
#      its id says missing; then, in every case, tagged raphael-journeys-app-base:<revision> by a separate call.
#   c  the platform image raphael-journeys-platform:<commit>: when image inspect of that reference says missing, the git
#      archive of the commit is shipped to S2dir.platform-<commit>.tar.part and renamed to S2dir.platform-<commit>.tar,
#      stack_up.sh build platform starts the build (it returns at once), and this script waits with the same inspect: at
#      once, then every 10 seconds (the last sleep cut to the time left), until the image is there or the box runs out.
#      A build still running at the box keeps running.
#   d  stack_up.sh build proxy; its tag is the last line "stack_up: built proxy <tag>".
#   e  stack_up.sh converge <proxy tag> <revision> <commit> <revision>.
#   f  /opt/raphael-page-check of the owner app container is copied into the browser container as
#      /opt/raphael-journeys/tools/page-check: one stage call (docker cp through S2dir.page-check.tar into page-check.new,
#      checked to hold something), then one swap call (the final folder to .old, .new to the final name; put back on error).
#   g  retention, in every run that gets this far (also when the stack was current): of each of the five repositories
#      the two newest tags stay, and so does every tag a container of the stack uses (stack_up.sh versions, read again);
#      every other tag is removed, one docker image rm of repo:tag at a time.
# A transfer writes only under a new name, every side of it (the producer on Server 1, the receiver on Server 2) is read
# through settle, and only then a separate call publishes (marker and renames, tag, final archive name, swap). A failed
# run never stops, removes or changes a running container, a volume or the active stack. It may leave staged files, an
# image without a tag, S2dir.previous and page-check .new or .old behind; a later run replaces them by exact name.
#
# Cleanup is the one place where this run removes files: the EXIT trap, once, at every end of the run (success, already
# current, any failure, the box). After step g it makes one ssh call for each flag that is set and removes this run's own
# files by exact name: the platform archive and its .part file (step c started a shipment; phase platform-build), the page
# check .old and S2dir.page-check.tar (step f was reached; phase page-check). Each call has its own bound of
# CLEANUP_SECONDS. A call that fails prints one line on stderr, "refresh_stack: cleanup failed: <phase> (<its reason>)", and
# changes neither the exit status, nor the convergence, nor the success line. Nothing was made by a run that is current, so
# it makes no cleanup call. Leftovers of earlier runs are not looked for: they are accepted.
#
# Output: on success one line on stdout, "refresh_stack: the stack is at the released versions". A failure ends with
# one line on stderr, "refresh_stack: <reason>" (the reasons are in reason_for), and exit 1. No line carries a value;
# the output of ssh, docker and git is parsed here and never echoed. Never used on Server 2: -f, --force, prune,
# docker rm, container rm, volume rm.
set -uo pipefail
umask 077
export LC_ALL=C TZ=UTC

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TEMPLATE=$HERE/stack.env.template
SSH_RE='^[A-Za-z0-9_][A-Za-z0-9_.@-]*$'
DIR_RE='^/[A-Za-z0-9_./-]+$'
NAME_RE='^[A-Za-z0-9][A-Za-z0-9_.-]*$'
TAG_RE='^[A-Za-z0-9._-]{1,63}$'
SHA_RE='^[0-9a-f]{40}$'
BOX_SECONDS=1500
CLEANUP_SECONDS=5
POLL_SECONDS=10
KEEP_TAGS=2
OPT_PAGE_CHECK=/opt/raphael-page-check
BROWSER_PAGE_CHECK=/opt/raphael-journeys/tools/page-check
SUCCESS_LINE='refresh_stack: the stack is at the released versions'
MADE_ARCHIVE=0 MADE_PAGE=0

reason_for() {
  case $1 in
    usage) echo "no argument is accepted" ;;
    config) echo "a setting or the template is missing or is not plain" ;;
    commit) echo "the platform commit could not be read" ;;
    gateway) echo "the gateway code version could not be read or is absent" ;;
    differ) echo "the platform commit and the gateway code version differ" ;;
    owner) echo "the owner app could not be read" ;;
    revision) echo "the owner app revision is not 40 lowercase hex" ;;
    ship-scripts) echo "the scripts folder could not be shipped to Server 2" ;;
    marker-ask) echo "Server 2 could not be asked which scripts folder it holds" ;;
    image-ask) echo "Server 2 could not be asked about an image" ;;
    app-image) echo "the owner app image could not be loaded or tagged on Server 2" ;;
    platform-build) echo "the platform build could not be shipped or started" ;;
    box) echo "the 25-minute box ran out; a platform build that is running keeps running" ;;
    proxy-tag) echo "the proxy tag was not reported" ;;
    converge) echo "the stack could not be converged" ;;
    page-check) echo "the page check could not be copied into the browser container" ;;
    prune) echo "old images could not be listed or removed" ;;
  esac
}

fail() {
  printf 'refresh_stack: %s\n' "$(reason_for "$1")" >&2
  exit 1
}

# cleanup: the one place where this run removes files, as the EXIT trap, so every end of the run passes through it once: the
# success line, the already-current run, any fail, the box. Each flag that is set gives one best-effort ssh call that removes
# this run's own files by exact name. Every call has its own bound of CLEANUP_SECONDS (not the time left of the box, which
# may be over), so the box is overrun by a few seconds at most. It never calls fail, budget or settle and never exits, so it
# changes neither the exit status nor the convergence. A call that fails prints one line on stderr that names its phase.
# A flag is set only after the names its call uses (PLATFORM_ARCHIVE, PAGE_TAR) are defined. Leftovers of earlier runs are
# not looked for. Never force, never a container or a volume.
PAGE_REMOVE='if [ -e "$1" ]; then rm -r "$1"; fi'
tidy() {
  timeout "$CLEANUP_SECONDS" ssh "$RUNNER" "$2" > /dev/null 2>&1 < /dev/null || printf 'refresh_stack: cleanup failed: %s (%s)\n' "$1" "$(reason_for "$1")" >&2
}
cleanup() {
  trap - EXIT
  if ((MADE_ARCHIVE == 1)); then
    tidy platform-build "r=0; if [ -e '$PLATFORM_ARCHIVE.part' ]; then rm '$PLATFORM_ARCHIVE.part' || r=1; fi; if [ -e '$PLATFORM_ARCHIVE' ]; then rm '$PLATFORM_ARCHIVE' || r=1; fi; exit \$r"
  fi
  if ((MADE_PAGE == 1)); then
    tidy page-check "docker exec '$BROWSER_C' sh -c '$PAGE_REMOVE' sh '$BROWSER_PAGE_CHECK.old' && r=0 || r=1; if [ -e '$PAGE_TAR' ]; then rm '$PAGE_TAR' || r=1; fi; exit \$r"
  fi
  return 0
}
trap cleanup EXIT

((${#} == 0)) || fail usage

# The settings, checked before anything is read or run.
RUNNER=${JOURNEY_RUNNER_SSH-}
S2_DIR=${JOURNEY_S2_SCRIPTS_DIR:-/home/raphael-runner/raphael-journeys}
while [[ $S2_DIR == */ ]]; do S2_DIR=${S2_DIR%/}; done
[[ $RUNNER =~ $SSH_RE ]] && [[ $S2_DIR =~ $DIR_RE ]] || fail config

# The names of the stack: the five containers in the order of their roles, and the one production container.
CONTAINERS=() PRODUCTION=()
read_template() {
  local line section="" section_re='^\[([^]]+)\]$'
  [[ -f $TEMPLATE && -r $TEMPLATE ]] || return 1
  while IFS= read -r line || [[ -n $line ]]; do
    line=${line#"${line%%[![:space:]]*}"}
    line=${line%"${line##*[![:space:]]}"}
    if [[ -z $line || $line == "#"* ]]; then
      continue
    fi
    if [[ $line =~ $section_re ]]; then
      section=${BASH_REMATCH[1]}
    elif [[ -z $section ]]; then
      return 1
    elif [[ $line =~ ^[A-Za-z0-9_.-]+$ ]]; then
      case $section in
        containers) CONTAINERS+=("$line") ;;
        production-container) PRODUCTION+=("$line") ;;
      esac
    else
      return 1
    fi
  done < "$TEMPLATE"
  local name
  ((${#CONTAINERS[@]} == 5 && ${#PRODUCTION[@]} == 1)) || return 1
  for name in "${CONTAINERS[@]}" "${PRODUCTION[0]}"; do
    [[ $name =~ $NAME_RE ]] || return 1
  done
}
read_template || fail config
APP_C=${CONTAINERS[1]}
BROWSER_C=${CONTAINERS[4]}
PRODUCTION_C=${PRODUCTION[0]}

START=$(date +%s 2> /dev/null < /dev/null)

# The box. budget puts the whole seconds left into LEFT (at least 1); with no time left, or a clock that cannot be read, the
# run ends with reason box and the call is not started. Every git, docker and ssh call of this script is
# timeout "$LEFT" <command> right after budget, and its status (every PIPESTATUS element of a pipeline) goes through settle.
# These functions end the run with fail, so they are only called in the main shell, never inside $( ... ).
LEFT=0
budget() {
  local now
  now=$(date +%s 2> /dev/null < /dev/null)
  [[ $START =~ ^[0-9]+$ && $now =~ ^[0-9]+$ ]] || fail box
  LEFT=$((BOX_SECONDS - (10#$now - 10#$START)))
  ((LEFT >= 1)) || fail box
}

# settle STATUS...: a status of 124 or 137 (ended by its budget) is box. So is a clock at or past the box, also when the
# call succeeded: a call that comes back late is late. Otherwise it returns the first status that is not 0.
settle() {
  local rc first=0 now
  for rc in "$@"; do
    if [[ $rc == 124 || $rc == 137 ]]; then fail box; fi
    if ((first == 0)) && [[ $rc != 0 ]]; then first=$rc; fi
  done
  now=$(date +%s 2> /dev/null < /dev/null)
  [[ $START =~ ^[0-9]+$ && $now =~ ^[0-9]+$ ]] || fail box
  ((10#$now - 10#$START < BOX_SECONDS)) || fail box
  return "$first"
}

# run_out COMMAND...: a command on Server 1 with the time that is left. Its output goes to RUN_OUT (parsed, never printed).
RUN_OUT=""
run_out() {
  local rc
  budget
  RUN_OUT=$(timeout "$LEFT" "$@" 2> /dev/null < /dev/null)
  rc=$?
  settle "$rc"
}

# The platform version: the commit of the checkout that holds this script.
run_out git -C "$HERE" rev-parse --show-toplevel || fail commit
ROOT=$RUN_OUT
[[ $ROOT == /* ]] || fail commit
run_out git -C "$ROOT" rev-parse HEAD || fail commit
COMMIT=$RUN_OUT
[[ $COMMIT =~ $SHA_RE ]] || fail commit

# It must be the version the gateway on this server runs.
GATEWAY_STATE=${HERMES_HOME:-${HOME-}/.hermes}/gateway_state.json
read_gateway_version() {
  local text
  [[ -f $GATEWAY_STATE && -r $GATEWAY_STATE ]] || return 1
  { text=$(< "$GATEWAY_STATE"); } 2> /dev/null || return 1
  [[ $text =~ \"code_sha\"[[:space:]]*:[[:space:]]*\"([0-9a-f]{40})\" ]] || return 1
  GATEWAY_SHA=${BASH_REMATCH[1]}
}
read_gateway_version || fail gateway
[[ $GATEWAY_SHA == "$COMMIT" ]] || fail differ

# The owner app version: the revision label and the image id of the production container, from one docker inspect
# whose format prints those two fields and nothing else.
run_out docker inspect --type container \
  --format '{{index .Config.Labels "org.opencontainers.image.revision"}} {{.Image}}' \
  "$PRODUCTION_C" || fail owner
OWNER=$RUN_OUT
[[ $OWNER =~ ^([^[:space:]]*)\ (sha256:[0-9a-f]{64})$ ]] || fail owner
REVISION=${BASH_REMATCH[1]}
IMAGE_ID=${BASH_REMATCH[2]}
[[ $REVISION =~ $SHA_RE ]] || fail revision

# remote COMMAND: one command on Server 2 through the runner identity, with the time that is left. Its output is dropped
# and it reads no input. It returns what settle returns.
remote() {
  local rc
  budget
  timeout "$LEFT" ssh "$RUNNER" "$1" > /dev/null 2>&1 < /dev/null
  rc=$?
  settle "$rc"
}

# remote_out COMMAND: the same, but the output is kept in REMOTE_OUT (parsed, never printed) and the exit status in
# REMOTE_RC. The sentinel keeps the trailing newlines of the output, which $(...) would strip.
REMOTE_OUT="" REMOTE_RC=1
remote_out() {
  local raw
  budget
  raw=$(
    timeout "$LEFT" ssh "$RUNNER" "$1" 2> /dev/null < /dev/null
    echo "x$?"
  )
  REMOTE_RC=${raw##*x}
  REMOTE_OUT=${raw%x*}
  settle "$REMOTE_RC"
}

# stream REMOTE_COMMAND PRODUCER...: the output of a command on Server 1 is the input of one command on Server 2. Both
# sides get the time that is left and all the statuses of the pipeline go through settle.
stream() {
  local remote_command=$1 rcs
  shift
  budget
  timeout "$LEFT" "$@" 2> /dev/null < /dev/null | timeout "$LEFT" ssh "$RUNNER" "$remote_command" > /dev/null 2>&1
  rcs=("${PIPESTATUS[@]}")
  settle "${rcs[@]}"
}

# image_state NAME: the one question about an image on Server 2 (NAME is an image id or the platform reference, both
# validated before), asked with one docker image inspect. 0 found, 1 missing, 2 could not be read. Found: status 0 and
# one line, sha256: and 64 lowercase hex (for an id, the id asked). Missing: status 1 and, after one optional leading empty
# line or one "[]" line, one line that is Docker's own not-found answer for exactly NAME. Everything else is 2. The answer
# is parsed here and never printed. A tag on the image does not matter.
image_state() {
  local name=$1 text line id_re='^sha256:[0-9a-f]{64}$'
  remote_out "docker image inspect --format '{{.Id}}' '$name' 2>&1"
  text=$REMOTE_OUT
  case $REMOTE_RC in
    0)
      line=${text%$'\n'}
      [[ $line =~ $id_re ]] || return 2
      if [[ $name == sha256:* && $line != "$name" ]]; then return 2; fi
      return 0
      ;;
    1)
      if [[ $text == $'\n'* ]]; then
        text=${text#$'\n'}
      elif [[ $text == '[]'$'\n'* ]]; then
        text=${text#'[]'$'\n'}
      fi
      line=${text%$'\n'}
      if [[ $line == "Error response from daemon: No such image: $name" || $line == "Error: No such image: $name" ]]; then
        return 1
      fi
      ;;
  esac
  return 2
}

# parse_versions: the answer of stack_up.sh versions (REMOTE_RC and REMOTE_OUT) read into VERSION_TAGS: the tags of proxy,
# app, dashboard, gateway and browser, in that order ("none": that container does not exist). It returns 1 when the call
# failed, when there are not exactly five lines "<role> <tag>" in that order, or when a tag is not plain. Steps 0 and g
# both use it.
VERSION_ROLES=(proxy app dashboard gateway browser)
VERSION_TAGS=()
parse_versions() {
  local lines=() tags=() i tag
  [[ $REMOTE_RC == 0 ]] || return 1
  mapfile -t lines <<< "${REMOTE_OUT%$'\n'}"
  ((${#lines[@]} == 5)) || return 1
  for i in 0 1 2 3 4; do
    [[ ${lines[i]} == "${VERSION_ROLES[i]} "* ]] || return 1
    tag=${lines[i]#"${VERSION_ROLES[i]} "}
    [[ $tag =~ $TAG_RE ]] || return 1
    tags+=("$tag")
  done
  VERSION_TAGS=("${tags[@]}")
}

# Step g, last (defined here because step 0 reaches it at once when the stack is current; steps a to f reach it at the end).
# The tags the five containers use come from its own stack_up.sh versions call. docker lists the newest image first, so per
# repository the first KEEP_TAGS plain tags stay, and so does every plain tag that one of the five containers uses; each
# other plain tag is removed by its exact reference, one call each. <none> and tags that are not plain are never counted or
# touched. The first call that fails (versions, list or removal) ends the run. Nothing else is exempt.
# The table holds each repository and the places in VERSION_TAGS of the containers made from it (the app base has none).
PRUNE_TABLE=(
  "raphael-journeys-proxy 0"
  "raphael-journeys-app-base"
  "raphael-journeys-app 1"
  "raphael-journeys-platform 2 3"
  "raphael-journeys-browser 4"
)
prune_repository() {
  local repo=$1 used=" " tag seen=0 tags i
  shift
  for i in "$@"; do
    if [[ ${VERSION_TAGS[i]} != none ]]; then used+="${VERSION_TAGS[i]} "; fi
  done
  remote_out "docker image ls '$repo' --format '{{.Tag}}'"
  [[ $REMOTE_RC == 0 ]] || fail prune
  tags=$REMOTE_OUT
  while IFS= read -r tag; do
    [[ $tag =~ $TAG_RE ]] || continue
    seen=$((seen + 1))
    if ((seen <= KEEP_TAGS)) || [[ $used == *" $tag "* ]]; then continue; fi
    remote "docker image rm '$repo:$tag'" || fail prune
  done <<< "$tags"
}
retention() {
  local row fields
  remote_out "cd '$S2_DIR' && ./stack_up.sh versions"
  parse_versions || fail prune
  for row in "${PRUNE_TABLE[@]}"; do
    read -ra fields <<< "$row"
    prune_repository "${fields[@]}"
  done
}

# finish: the end of every run that gets this far: step g, then the success line.
finish() {
  retention
  echo "$SUCCESS_LINE"
  exit 0
}

# Step 0. The first look: three read-only calls in this order, each only when the one before agreed with the released
# versions. The stack is current only when all three agree. MARKER_SAME is 1 when the first agreed (the scripts folder on
# Server 2 came from this commit): step a then ships no scripts. The first call answers 3 when there is positive evidence
# that there is no marker (the folder can be searched and has no such entry, or there is no folder and its parent can be
# searched), and 4 when the path cannot be examined. On status 0 the output must be exactly one line of 40 lowercase hex
# digits (the final newline may be missing): this commit, or another commit, which is not current. Anything else is a read
# that failed and ends the run with marker-ask. The other two calls decide no transfer.
MARKER_SAME=0
S2_PARENT=${S2_DIR%/*}
if [[ -z $S2_PARENT ]]; then S2_PARENT=/; fi
S2_MARKER=$S2_DIR/.refresh-commit
marker_cmd="if [ -d '$S2_DIR' ] && [ -x '$S2_DIR' ]; then if [ -e '$S2_MARKER' ] || [ -L '$S2_MARKER' ]; then cat '$S2_MARKER'; else exit 3; fi;"
marker_cmd+=" elif [ -d '$S2_PARENT' ] && [ -x '$S2_PARENT' ] && [ ! -e '$S2_DIR' ] && [ ! -L '$S2_DIR' ]; then exit 3; else exit 4; fi"
stack_is_current() {
  local marker app_re="^$REVISION-[0-9a-f]{12}\$"
  remote_out "$marker_cmd"
  case $REMOTE_RC in
    0) ;;
    3) return 1 ;;
    *) fail marker-ask ;;
  esac
  marker=${REMOTE_OUT%$'\n'}
  [[ $marker =~ $SHA_RE ]] || fail marker-ask
  [[ $marker == "$COMMIT" ]] || return 1
  MARKER_SAME=1
  remote_out "cd '$S2_DIR' && ./stack_up.sh versions"
  parse_versions || return 1
  [[ ${VERSION_TAGS[0]} != none ]] || return 1
  [[ ${VERSION_TAGS[2]} == "$COMMIT" && ${VERSION_TAGS[3]} == "$COMMIT" ]] || return 1
  [[ ${VERSION_TAGS[1]} =~ $app_re && ${VERSION_TAGS[4]} =~ $app_re ]] || return 1
  remote "docker exec '$BROWSER_C' test -d '$BROWSER_PAGE_CHECK'"
}
if stack_is_current; then finish; fi

# Step a. scripts/journeys of the commit. The stage extracts the git archive into S2dir.new (the files directly in it; a
# stale .new is removed first) and touches nothing else, so the active folder and its marker stay as they were until the
# stream has settled with 0 for the producer, ssh and tar alike. The publish is a second call: the marker .refresh-commit
# (the commit this folder came from) is written into the new folder, the folder it replaces becomes S2dir.previous (one
# generation, a stale .previous is removed first) and the new folder is renamed to S2dir, so folder and marker become
# active together. A publish that fails at any point leaves S2dir with its old files and old marker: when the removal of a
# stale .previous or the first rename fails nothing has changed, and when the second rename fails the first is undone
# (S2dir.previous is renamed back to S2dir). With no S2dir (a first run) the one rename is the whole publish. Either way
# the run ends with ship-scripts, and a retry at the same commit ships again. When the marker on Server 2 already names
# this commit, no scripts archive is made at all.
S2_NEW=$S2_DIR.new
S2_PREVIOUS=$S2_DIR.previous
stage_scripts_cmd="if [ -e '$S2_NEW' ]; then rm -r '$S2_NEW'; fi && mkdir '$S2_NEW' && tar -xf - -C '$S2_NEW' --strip-components=2"
publish_scripts_cmd="echo '$COMMIT' > '$S2_NEW/.refresh-commit' && if [ -e '$S2_DIR' ]; then"
publish_scripts_cmd+=" if [ -e '$S2_PREVIOUS' ]; then rm -r '$S2_PREVIOUS'; fi && mv '$S2_DIR' '$S2_PREVIOUS'"
publish_scripts_cmd+=" && { mv '$S2_NEW' '$S2_DIR' || { mv '$S2_PREVIOUS' '$S2_DIR'; false; }; }"
publish_scripts_cmd+="; else mv '$S2_NEW' '$S2_DIR'; fi"
if ((MARKER_SAME == 0)); then
  stream "$stage_scripts_cmd" git -C "$ROOT" archive --format=tar "$COMMIT" scripts/journeys || fail ship-scripts
  remote "$publish_scripts_cmd" || fail ship-scripts
fi

# Step b. The owner app image: shipped only when Server 2 lacks that id, and tagged in every case. The stage is docker save
# into docker load (a loaded image id has no tag); the publish is the tag, a separate call that only starts when the
# stream has settled with 0 on both sides.
image_state "$IMAGE_ID"
case $? in
  0) ;;
  1) stream "docker load" docker save "$IMAGE_ID" || fail app-image ;;
  *) fail image-ask ;;
esac
remote "docker tag '$IMAGE_ID' 'raphael-journeys-app-base:$REVISION'" || fail app-image

# Step c. The platform image. Its archive is shipped as one file only when the image is missing: the stage writes it under
# the name .part, and only when the stream has settled with 0 on both sides a second call renames it to the archive name,
# so no file with the archive name exists unless it is whole. Only then the build is started detached by stack_up.sh and
# waited for here: the first look at once, then every POLL_SECONDS (the last sleep cut to the time that is left), until
# the box runs out. A build still running then is left running; every call of the wait ends the run with reason box when
# it comes back at or after the box. The archive and its .part file are removed by cleanup at the end of the run.
PLATFORM_REF=raphael-journeys-platform:$COMMIT
PLATFORM_ARCHIVE=$S2_DIR.platform-$COMMIT.tar
build_platform() {
  local nap
  MADE_ARCHIVE=1
  stream "cat > '$PLATFORM_ARCHIVE.part'" git -C "$ROOT" archive --format=tar "$COMMIT" || fail platform-build
  remote "mv '$PLATFORM_ARCHIVE.part' '$PLATFORM_ARCHIVE'" || fail platform-build
  remote "cd '$S2_DIR' && ./stack_up.sh build platform '$COMMIT' '$PLATFORM_ARCHIVE'" || fail platform-build
  while :; do
    image_state "$PLATFORM_REF"
    case $? in
      0) break ;;
      1) ;;
      *) fail image-ask ;;
    esac
    budget
    nap=$POLL_SECONDS
    if ((LEFT < nap)); then nap=$LEFT; fi
    sleep "$nap" > /dev/null 2>&1 < /dev/null
  done
}
image_state "$PLATFORM_REF"
case $? in
  0) ;;
  1) build_platform ;;
  *) fail image-ask ;;
esac

# Step d. The proxy image. Its tag is the one the build reports; the last such line counts.
remote_out "cd '$S2_DIR' && ./stack_up.sh build proxy"
[[ $REMOTE_RC == 0 ]] || fail proxy-tag
PROXY_TAG=""
while IFS= read -r line; do
  if [[ $line == "stack_up: built proxy "* ]]; then PROXY_TAG=${line#"stack_up: built proxy "}; fi
done <<< "$REMOTE_OUT"
[[ $PROXY_TAG =~ $TAG_RE ]] || fail proxy-tag

# Step e. The stack is brought to the versions. It is done when converge exits 0 and its last non-empty line is one of
# the two lines that mean the stack is as it can be before the first reset.
remote_out "cd '$S2_DIR' && ./stack_up.sh converge '$PROXY_TAG' '$REVISION' '$COMMIT' '$REVISION'"
[[ $REMOTE_RC == 0 ]] || fail converge
LAST_LINE=""
while IFS= read -r line; do
  if [[ -n $line ]]; then LAST_LINE=$line; fi
done <<< "$REMOTE_OUT"
[[ $LAST_LINE == "stack_up: the stack answers" ||
  $LAST_LINE == "stack_up: the first reset starts the owner app and the gateway" ]] || fail converge

# Step f. The page check of the owner app, copied container to container on Server 2 (docker cp reads a stopped owner
# app too), staged and published in two remote calls. The stage is one command with the steps joined by && (the remote
# shell has no pipefail, so no pipe joins the two docker cp): the copy is made beside the final folder (.new, after a
# stale .new or .old is removed), out of the owner app into the staging file S2dir.page-check.tar and out of that file
# into the browser container, and then .new is checked to hold something (an empty stream is not an error for docker cp).
# Only when the stage has reported 0 does a second call swap .new in with renames (the old folder is kept as .old and put
# back when the rename of .new did not work). A step that fails ends the run before the swap, so the final folder is never
# half a copy and stays as it was. The staging file and .old are removed by cleanup at the end of the run.
PAGE_TAR=$S2_DIR.page-check.tar
PAGE_PREPARE='for d in "$1" "$2"; do if [ -e "$d" ]; then rm -r "$d" || exit 1; fi; done; mkdir -p "$1"'
PAGE_CHECK='set -- "$1"/*; [ -e "$1" ] || exit 1'
PAGE_SWAP='n=$1; f=$2; o=$3; if [ -e "$f" ]; then mv "$f" "$o" || exit 1; fi;'
PAGE_SWAP+=' if mv "$n" "$f"; then exit 0; fi; if [ -e "$o" ]; then mv "$o" "$f"; fi; exit 1'
page_check_stage="docker exec '$BROWSER_C' sh -c '$PAGE_PREPARE' sh '$BROWSER_PAGE_CHECK.new' '$BROWSER_PAGE_CHECK.old'"
page_check_stage+=" && docker cp '$APP_C:$OPT_PAGE_CHECK/.' - > '$PAGE_TAR'"
page_check_stage+=" && docker cp - '$BROWSER_C:$BROWSER_PAGE_CHECK.new' < '$PAGE_TAR'"
page_check_stage+=" && docker exec '$BROWSER_C' sh -c '$PAGE_CHECK' sh '$BROWSER_PAGE_CHECK.new'"
MADE_PAGE=1
remote "$page_check_stage" || fail page-check
remote "docker exec '$BROWSER_C' sh -c '$PAGE_SWAP' sh '$BROWSER_PAGE_CHECK.new' '$BROWSER_PAGE_CHECK' '$BROWSER_PAGE_CHECK.old'" || fail page-check

# Step g, last: finish runs retention (defined above) and ends the run; cleanup follows it.
finish
