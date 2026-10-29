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
# Steps, in this order (S2dir is the Server 2 folder):
#   0  read-only first look: stack_up.sh versions and the page check folder of the browser container. When all of it is
#      at the released versions nothing is shipped, built, converged or pruned. A versions failure is not a failure here.
#   a  scripts/journeys of the commit is shipped as a git archive into S2dir.new and swapped in with two renames
#      (the old folder stays as S2dir.previous)
#   b  the owner app image is shipped (docker save into docker load) only when Server 2 lacks that id, then tagged
#      raphael-journeys-app-base:<revision>
#   c  the platform image raphael-journeys-platform:<commit>: when Server 2 lacks it, the git archive of the commit is
#      shipped as one file, stack_up.sh build platform starts the build (it returns at once) and this script waits for the
#      image: first look at once, then every 10 seconds, at most 25 minutes from the start of this script. Nothing on
#      Server 2 is ever killed or stopped; a build still running at the box keeps running.
#   d  stack_up.sh build proxy; its tag is read from the line "stack_up: built proxy <tag>"
#   e  stack_up.sh converge <proxy tag> <revision> <commit> <revision>
#   f  /opt/raphael-page-check of the owner app container is copied into the browser container as
#      /opt/raphael-journeys/tools/page-check (docker cp streams on Server 2 only; the old copy is replaced by renames)
#   g  last: of each of the five repositories the two newest tags stay and every other is removed, one
#      docker image rm of repo:tag at a time, except what this run just used
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
    box) echo "the platform build was still running at the 25-minute box and keeps running" ;;
    proxy-tag) echo "the proxy tag was not reported" ;;
    converge) echo "the stack could not be converged" ;;
    page-check) echo "the page check could not be copied into the browser container" ;;
    prune) echo "old images could not be removed" ;;
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

# The platform version: the commit of the checkout that holds this script.
ROOT=$(git -C "$HERE" rev-parse --show-toplevel 2> /dev/null < /dev/null) || fail commit
[[ $ROOT == /* ]] || fail commit
COMMIT=$(git -C "$ROOT" rev-parse HEAD 2> /dev/null < /dev/null) || fail commit
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
OWNER=$(docker inspect --type container \
  --format '{{index .Config.Labels "org.opencontainers.image.revision"}} {{.Image}}' \
  "$PRODUCTION_C" 2> /dev/null < /dev/null) || fail owner
[[ $OWNER =~ ^([^[:space:]]*)\ (sha256:[0-9a-f]{64})$ ]] || fail owner
REVISION=${BASH_REMATCH[1]}
IMAGE_ID=${BASH_REMATCH[2]}
[[ $REVISION =~ $SHA_RE ]] || fail revision

# remote COMMAND: one command on Server 2 through the runner identity. Its output is dropped and it reads no input.
remote() { ssh "$RUNNER" "$1" > /dev/null 2>&1 < /dev/null; }

# remote_out COMMAND: the same, but the output is kept in REMOTE_OUT (parsed, never printed) and the exit status in
# REMOTE_RC. The sentinel keeps the trailing newlines of the output, which $(...) would strip.
REMOTE_OUT="" REMOTE_RC=1
remote_out() {
  local raw
  raw=$(
    ssh "$RUNNER" "$1" 2> /dev/null < /dev/null
    echo "x$?"
  )
  REMOTE_RC=${raw##*x}
  REMOTE_OUT=${raw%x*}
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

# Step 0. The stack is current only when every part of the first look agrees with the released versions.
stack_is_current() {
  local roles=(proxy app dashboard gateway browser) lines=() i tag app_re="^$REVISION-[0-9a-f]{12}\$"
  remote_out "cd '$S2_DIR' && ./stack_up.sh versions"
  [[ $REMOTE_RC == 0 ]] || return 1
  mapfile -t lines <<< "${REMOTE_OUT%$'\n'}"
  ((${#lines[@]} == 5)) || return 1
  for i in 0 1 2 3 4; do
    [[ ${lines[i]} == "${roles[i]} "* ]] || return 1
    tag=${lines[i]#"${roles[i]} "}
    case ${roles[i]} in
      proxy) [[ $tag =~ $TAG_RE && $tag != none ]] || return 1 ;;
      dashboard | gateway) [[ $tag == "$COMMIT" ]] || return 1 ;;
      app | browser) [[ $tag =~ $app_re ]] || return 1 ;;
    esac
  done
  remote "docker exec '$BROWSER_C' test -d '$BROWSER_PAGE_CHECK'"
}
if stack_is_current; then
  echo "$SUCCESS_LINE"
  exit 0
fi

# Step a. scripts/journeys of the commit goes into S2dir.new (the files directly in it) and is swapped in with two
# renames; the folder it replaces stays as S2dir.previous, one generation. A stale .new or .previous is removed first.
S2_NEW=$S2_DIR.new
S2_PREVIOUS=$S2_DIR.previous
ship_scripts_cmd="if [ -e '$S2_NEW' ]; then rm -r '$S2_NEW'; fi && mkdir '$S2_NEW' && tar -xf - -C '$S2_NEW' --strip-components=2"
ship_scripts_cmd+=" && if [ -e '$S2_DIR' ]; then if [ -e '$S2_PREVIOUS' ]; then rm -r '$S2_PREVIOUS'; fi && mv '$S2_DIR' '$S2_PREVIOUS'; fi"
ship_scripts_cmd+=" && mv '$S2_NEW' '$S2_DIR'"
git -C "$ROOT" archive --format=tar "$COMMIT" scripts/journeys 2> /dev/null < /dev/null |
  ssh "$RUNNER" "$ship_scripts_cmd" > /dev/null 2>&1 || fail ship-scripts

# Step b. The owner app image: shipped only when Server 2 lacks that id, and tagged in every case.
image_known "$IMAGE_ID"
case $? in
  0) ;;
  1) docker save "$IMAGE_ID" 2> /dev/null < /dev/null | ssh "$RUNNER" "docker load" > /dev/null 2>&1 || fail app-image ;;
  *) fail image-ask ;;
esac
remote "docker tag '$IMAGE_ID' 'raphael-journeys-app-base:$REVISION'" || fail app-image

# Step c. The platform image. Its archive is shipped as one file (a temporary name, then renamed) only when the image is
# missing; the build is started detached by stack_up.sh and waited for here: the first look at once, then every
# POLL_SECONDS, until BOX_SECONDS have passed since this script started. A build still running then is left running.
PLATFORM_REF=raphael-journeys-platform:$COMMIT
PLATFORM_ARCHIVE=$S2_DIR.platform-$COMMIT.tar
build_platform() {
  local now
  git -C "$ROOT" archive --format=tar "$COMMIT" 2> /dev/null < /dev/null |
    ssh "$RUNNER" "cat > '$PLATFORM_ARCHIVE.part' && mv '$PLATFORM_ARCHIVE.part' '$PLATFORM_ARCHIVE'" > /dev/null 2>&1 ||
    fail platform-build
  remote "cd '$S2_DIR' && ./stack_up.sh build platform '$COMMIT' '$PLATFORM_ARCHIVE'" || fail platform-build
  while :; do
    image_known "$PLATFORM_REF"
    case $? in
      0) break ;;
      1) ;;
      *) fail image-ask ;;
    esac
    now=$(date +%s 2> /dev/null < /dev/null)
    [[ $START =~ ^[0-9]+$ && $now =~ ^[0-9]+$ ]] || fail box
    ((now - START < BOX_SECONDS)) || fail box
    sleep "$POLL_SECONDS" > /dev/null 2>&1 < /dev/null
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
# app too). The copy is made beside the final folder (.new, after a stale .new or .old is removed), checked to hold
# something (an empty stream is not an error for docker cp), and swapped in with renames; the old folder is removed
# last. A copy that fails stops before the swap, so the final folder is never half a copy.
PAGE_PREPARE='for d in "$1" "$2"; do if [ -e "$d" ]; then rm -r "$d" || exit 1; fi; done; mkdir -p "$1"'
PAGE_SWAP='n=$1; f=$2; o=$3; set -- "$n"/*; [ -e "$1" ] || exit 1; if [ -e "$f" ]; then mv "$f" "$o" || exit 1; fi;'
PAGE_SWAP+=' if mv "$n" "$f"; then if [ -e "$o" ]; then rm -r "$o"; fi; exit 0; fi; if [ -e "$o" ]; then mv "$o" "$f"; fi; exit 1'
page_check_cmd="docker exec '$BROWSER_C' sh -c '$PAGE_PREPARE' sh '$BROWSER_PAGE_CHECK.new' '$BROWSER_PAGE_CHECK.old'"
page_check_cmd+=" && docker cp '$APP_C:$OPT_PAGE_CHECK/.' - | docker cp - '$BROWSER_C:$BROWSER_PAGE_CHECK.new'"
page_check_cmd+=" && docker exec '$BROWSER_C' sh -c '$PAGE_SWAP' sh '$BROWSER_PAGE_CHECK.new' '$BROWSER_PAGE_CHECK' '$BROWSER_PAGE_CHECK.old'"
remote "$page_check_cmd" || fail page-check

# Step g, last. Per repository: the table holds the repository and the tag pattern this run just used there (the
# proxy tag, the app base revision, the platform commit, and in the app and browser repositories every tag that starts
# with the revision). Those stay whatever their age. docker lists the newest image first, so the first KEEP_TAGS plain
# tags stay and each other plain tag is removed by its exact reference, one call each. <none> and tags that are not
# plain are never touched. The first removal that fails ends the run.
PRUNE_TABLE=(
  "raphael-journeys-proxy $PROXY_TAG"
  "raphael-journeys-app-base $REVISION"
  "raphael-journeys-app $REVISION-*"
  "raphael-journeys-platform $COMMIT"
  "raphael-journeys-browser $REVISION-*"
)
prune_repository() {
  local repo=$1 used=$2 tag seen=0
  remote_out "docker image ls '$repo' --format '{{.Tag}}'"
  [[ $REMOTE_RC == 0 ]] || fail prune
  local tags=$REMOTE_OUT
  while IFS= read -r tag; do
    [[ $tag =~ $TAG_RE ]] || continue
    seen=$((seen + 1))
    # shellcheck disable=SC2053  # $used is a glob on purpose
    if ((seen <= KEEP_TAGS)) || [[ $tag == $used ]]; then continue; fi
    remote "docker image rm '$repo:$tag'" || fail prune
  done <<< "$tags"
}
for row in "${PRUNE_TABLE[@]}"; do
  read -r repository used_tags <<< "$row"
  prune_repository "$repository" "$used_tags"
done

echo "$SUCCESS_LINE"
exit 0
