#!/usr/bin/env bash
# Bring the journeys test stack on Server 2 to the released versions (card 3). Runs on Server 1, takes no argument,
# and exits 0 only when the stack is at the released versions.
#
# The released versions:
#   platform  the commit of the checkout that holds this script (git, found from this script's own folder). It must
#             equal code_sha in HERMES_HOME/gateway_state.json (read here without jq or python): the gateway on
#             Server 1 runs the same commit. Different, unreadable or absent: refused before any Server 2 step.
#   owner app the org.opencontainers.image.revision label and the image id of the production container named in
#             [production-container] of stack.env.template. One docker inspect with a two-field format reads them.
#             On Server 1 this script runs only that docker inspect and docker save of that image id; it changes nothing.
#
# Settings (the operator's environment):
#   JOURNEY_RUNNER_SSH         ssh destination of the runner identity on Server 2 (required; same rule as run_once.sh)
#   JOURNEY_S2_SCRIPTS_DIR     Server 2 folder that holds scripts/journeys of the released commit (same rule and default
#                              as run_once.sh; a trailing slash is stripped; the root is refused)
#   HERMES_HOME                Server 1 platform home. Default ~/.hermes
# Container names come from stack.env.template next to this script, read at run time. Only the five image repository
# names are typed in this script. Every Server 2 step is one ssh "$JOURNEY_RUNNER_SSH" "<command>" with exactly two
# arguments; a command that runs a script of the folder starts with cd '<folder>' && . No command line carries a value.
#
# The box is 25 minutes from the start of this script. Every git, docker and ssh call, both sides of a pipeline included,
# runs as timeout <seconds left> <command>; with no time left the call is not started. The clock is read before and after
# each call: a call that comes back at or after the box is late even when it succeeded. A status of 124 or 137, or a
# failure when the box is over, ends the run with reason box. Nothing on Server 2 is killed, stopped or removed for it.
#
# Steps, in this order (S2dir is the Server 2 folder):
#   0  read-only first look, three ssh calls, each only when the one before agreed: S2dir/.refresh-commit holds the
#      commit, stack_up.sh versions shows the released versions, the page check folder exists in the browser container.
#      When all three agree nothing is shipped, built or converged and only step g runs. Anything else means not current.
#   a  scripts/journeys of the commit is shipped as a git archive into S2dir.new, with the file .refresh-commit (the
#      commit) written into it, and swapped in with two renames (the old folder stays as S2dir.previous). Only when
#      S2dir/.refresh-commit differs from the commit.
#   b  the owner app image is shipped (docker save into docker load) only when Server 2 lacks that id, then tagged
#      raphael-journeys-app-base:<revision>
#   c  the platform image raphael-journeys-platform:<commit>: when Server 2 lacks it, the git archive of the commit is
#      shipped as one file, stack_up.sh build platform starts the build (it returns at once) and this script waits for the
#      image: first look at once, then every 10 seconds (the last sleep cut to the time that is left), until the box
#      runs out. Nothing on Server 2 is ever killed or stopped; a build still running at the box keeps running.
#   d  stack_up.sh build proxy; its tag is read from the line "stack_up: built proxy <tag>"
#   e  stack_up.sh converge <proxy tag> <revision> <commit> <revision>
#   f  /opt/raphael-page-check of the owner app container is copied into the browser container as
#      /opt/raphael-journeys/tools/page-check (docker cp streams on Server 2 only, through the file S2dir.page-check.tar;
#      the old copy is replaced by renames, and only after both copies succeeded)
#   g  last, in every run that gets this far (also when the stack was already current): of each of the five repositories
#      the two newest tags stay, and so does every tag that one of the five containers uses (stack_up.sh versions, read
#      again); every other tag is removed, one docker image rm of repo:tag at a time
# Never used on Server 2: -f, --force, prune, docker rm, container rm, volume rm.
#
# Output: on success one line on stdout, "refresh_stack: the stack is at the released versions". A failure ends with
# exactly one line on stderr, "refresh_stack: <reason>" (the reasons are in reason_for), and exit 1. No output line carries
# a value, and the output of ssh, docker and git is parsed here and never echoed.
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
POLL_SECONDS=10
KEEP_TAGS=2
OPT_PAGE_CHECK=/opt/raphael-page-check
BROWSER_PAGE_CHECK=/opt/raphael-journeys/tools/page-check
SUCCESS_LINE='refresh_stack: the stack is at the released versions'

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

# image_known REFERENCE: 0 when Server 2 holds the image, 1 when it does not, 2 when it could not be asked.
image_known() {
  remote "docker image inspect '$1'"
  case $? in
    0) return 0 ;;
    1) return 1 ;;
    *) return 2 ;;
  esac
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
# Server 2 came from this commit): step a then ships no scripts.
MARKER_SAME=0
stack_is_current() {
  local marker app_re="^$REVISION-[0-9a-f]{12}\$"
  remote_out "cat '$S2_DIR/.refresh-commit'"
  marker=${REMOTE_OUT%$'\n'}
  [[ $REMOTE_RC == 0 && $marker == "$COMMIT" ]] || return 1
  MARKER_SAME=1
  remote_out "cd '$S2_DIR' && ./stack_up.sh versions"
  parse_versions || return 1
  [[ ${VERSION_TAGS[0]} != none ]] || return 1
  [[ ${VERSION_TAGS[2]} == "$COMMIT" && ${VERSION_TAGS[3]} == "$COMMIT" ]] || return 1
  [[ ${VERSION_TAGS[1]} =~ $app_re && ${VERSION_TAGS[4]} =~ $app_re ]] || return 1
  remote "docker exec '$BROWSER_C' test -d '$BROWSER_PAGE_CHECK'"
}
if stack_is_current; then finish; fi

# Step a. scripts/journeys of the commit goes into S2dir.new (the files directly in it) and is swapped in with two
# renames; the folder it replaces stays as S2dir.previous, one generation. A stale .new or .previous is removed first.
# The marker .refresh-commit (the commit this folder came from) is written into the new folder after tar and before the
# renames, so the swap publishes folder and marker together. When the marker on Server 2 already names this commit,
# no scripts archive is made at all.
S2_NEW=$S2_DIR.new
S2_PREVIOUS=$S2_DIR.previous
ship_scripts_cmd="if [ -e '$S2_NEW' ]; then rm -r '$S2_NEW'; fi && mkdir '$S2_NEW' && tar -xf - -C '$S2_NEW' --strip-components=2"
ship_scripts_cmd+=" && echo '$COMMIT' > '$S2_NEW/.refresh-commit'"
ship_scripts_cmd+=" && if [ -e '$S2_DIR' ]; then if [ -e '$S2_PREVIOUS' ]; then rm -r '$S2_PREVIOUS'; fi && mv '$S2_DIR' '$S2_PREVIOUS'; fi"
ship_scripts_cmd+=" && mv '$S2_NEW' '$S2_DIR'"
if ((MARKER_SAME == 0)); then
  stream "$ship_scripts_cmd" git -C "$ROOT" archive --format=tar "$COMMIT" scripts/journeys || fail ship-scripts
fi

# Step b. The owner app image: shipped only when Server 2 lacks that id, and tagged in every case.
image_known "$IMAGE_ID"
case $? in
  0) ;;
  1) stream "docker load" docker save "$IMAGE_ID" || fail app-image ;;
  *) fail image-ask ;;
esac
remote "docker tag '$IMAGE_ID' 'raphael-journeys-app-base:$REVISION'" || fail app-image

# Step c. The platform image. Its archive is shipped as one file (a temporary name, then renamed) only when the image is
# missing; the build is started detached by stack_up.sh and waited for here: the first look at once, then every
# POLL_SECONDS (the last sleep cut to the time that is left), until the box runs out. A build still running then is left
# running; every call of the wait ends the run with reason box when it comes back at or after the box.
PLATFORM_REF=raphael-journeys-platform:$COMMIT
PLATFORM_ARCHIVE=$S2_DIR.platform-$COMMIT.tar
build_platform() {
  local nap
  stream "cat > '$PLATFORM_ARCHIVE.part' && mv '$PLATFORM_ARCHIVE.part' '$PLATFORM_ARCHIVE'" \
    git -C "$ROOT" archive --format=tar "$COMMIT" || fail platform-build
  remote "cd '$S2_DIR' && ./stack_up.sh build platform '$COMMIT' '$PLATFORM_ARCHIVE'" || fail platform-build
  while :; do
    image_known "$PLATFORM_REF"
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
  # A file that cannot be removed is no reason to fail: the image is there and the next build replaces the file.
  remote "rm '$PLATFORM_ARCHIVE'"
}
image_known "$PLATFORM_REF"
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
# app too). One remote command with the steps joined by && (the remote shell has no pipefail, so no pipe joins the two
# docker cp): the copy is made beside the final folder (.new, after a stale .new or .old is removed), out of the owner app
# into the staging file S2dir.page-check.tar and out of that file into the browser container; only when both succeeded it
# is checked to hold something (an empty stream is not an error for docker cp) and swapped in with renames; the old folder
# is removed last, then the staging file. A step that fails stops the chain before the swap, so the final folder is never
# half a copy.
PAGE_PREPARE='for d in "$1" "$2"; do if [ -e "$d" ]; then rm -r "$d" || exit 1; fi; done; mkdir -p "$1"'
PAGE_SWAP='n=$1; f=$2; o=$3; set -- "$n"/*; [ -e "$1" ] || exit 1; if [ -e "$f" ]; then mv "$f" "$o" || exit 1; fi;'
PAGE_SWAP+=' if mv "$n" "$f"; then if [ -e "$o" ]; then rm -r "$o"; fi; exit 0; fi; if [ -e "$o" ]; then mv "$o" "$f"; fi; exit 1'
page_check_cmd="docker exec '$BROWSER_C' sh -c '$PAGE_PREPARE' sh '$BROWSER_PAGE_CHECK.new' '$BROWSER_PAGE_CHECK.old'"
page_check_cmd+=" && docker cp '$APP_C:$OPT_PAGE_CHECK/.' - > '$S2_DIR.page-check.tar'"
page_check_cmd+=" && docker cp - '$BROWSER_C:$BROWSER_PAGE_CHECK.new' < '$S2_DIR.page-check.tar'"
page_check_cmd+=" && docker exec '$BROWSER_C' sh -c '$PAGE_SWAP' sh '$BROWSER_PAGE_CHECK.new' '$BROWSER_PAGE_CHECK' '$BROWSER_PAGE_CHECK.old'"
page_check_cmd+=" && if [ -e '$S2_DIR.page-check.tar' ]; then rm '$S2_DIR.page-check.tar'; fi"
remote "$page_check_cmd" || fail page-check

# Step g, last: finish runs retention (defined above) and ends the run.
finish
