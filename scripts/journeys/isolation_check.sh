#!/usr/bin/env bash
# Isolation check of the journeys test stack (card P-2, item 4).
#   isolation_check.sh server2   proves the stack on Server 2 is isolated from production
#   isolation_check.sh server1   proves production's owner app names an issuer that is not the proxy's
# One line per check: "PASS <check>" or "FAIL <check>: <reason>". Exit 0 only when every check passed,
# 1 when any check failed or could not run, 2 for a usage or host error. No value is ever printed:
# messages name the check and the name concerned, never a value.
set -euo pipefail
umask 077
export LC_ALL=C

usage() {
  echo "usage: isolation_check.sh server2|server1" >&2
  exit 2
}

[ "$#" -eq 1 ] || usage
MODE=$1
case "$MODE" in
  server2 | server1) ;;
  *) usage ;;
esac

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TEMPLATE="$HERE/stack.env.template"

declare -A SECTION=()

read_template() {
  local line section=""
  [ -r "$TEMPLATE" ] || { echo "isolation_check: the template is not readable" >&2; exit 2; }
  while IFS= read -r line || [ -n "$line" ]; do
    line=${line#"${line%%[![:space:]]*}"}
    line=${line%"${line##*[![:space:]]}"}
    case "$line" in '' | '#'*) continue ;; esac
    if [[ $line =~ ^\[(.+)\]$ ]]; then
      section=${BASH_REMATCH[1]}
      SECTION[$section]=${SECTION[$section]-}
      continue
    fi
    [ -n "$section" ] || { echo "isolation_check: the template has a name outside a section" >&2; exit 2; }
    SECTION[$section]+="$line"$'\n'
  done < "$TEMPLATE"
}

# names_in SECTION: the names of one template section, one per line.
names_in() {
  printf '%s' "${SECTION[$1]-}"
}

# listed SECTION NAME: succeeds when NAME is on that section of the template.
listed() {
  local want=$2 name
  while IFS= read -r name; do
    [ "$name" = "$want" ] && return 0
  done < <(names_in "$1")
  return 1
}

read_template

# The ORDER of the template's [containers] is the role of each entry.
mapfile -t CONTAINERS < <(names_in containers)
mapfile -t PRODUCTION < <(names_in production-container)
mapfile -t PRODUCTION_VAR < <(names_in production-read-variable)
if [ "${#CONTAINERS[@]}" -ne 5 ] || [ "${#PRODUCTION[@]}" -ne 1 ] || [ "${#PRODUCTION_VAR[@]}" -ne 1 ]; then
  echo "isolation_check: the template does not have the expected container entries" >&2
  exit 2
fi
PROXY_C=${CONTAINERS[0]}
GATE_C=${CONTAINERS[3]}
PROD_C=${PRODUCTION[0]}
PROD_VAR=${PRODUCTION_VAR[0]}

FAILED=0

pass() {
  printf 'PASS %s\n' "$1"
}

# fail CHECK REASON: the reason names the check's subject, never a value.
fail() {
  local reason=${2//[$'\n\r']/ }
  printf 'FAIL %s: %s\n' "$1" "$reason"
  FAILED=1
}

# The stack host guard: this script only ever runs on the Server 2 host named by the operator.
require_stack_host() {
  local expected=${JOURNEY_STACK_HOST-} actual
  if [ -z "$expected" ]; then
    fail host "JOURNEY_STACK_HOST is not set"
    exit 1
  fi
  actual=$(hostname 2>/dev/null) || actual=""
  if [ "$actual" != "$expected" ]; then
    fail host "this is not the stack host"
    exit 1
  fi
}

# The only docker --format shapes this script uses.
F_IMAGE='{{.Image}}'
F_ENV_NAMES='{{range .Config.Env}}{{println (index (split . "=") 0)}}{{end}}'
F_MOUNT_TYPES='{{range .Mounts}}{{println .Type .Name}}{{end}}'

DK_OUT=""
DK_RC=0

# dk ARGS...: runs docker. Output goes to DK_OUT and the status to DK_RC; stderr is dropped
# because a container can print a value there.
dk() {
  DK_RC=0
  DK_OUT=$(docker "$@" 2>/dev/null) || DK_RC=$?
}

PROBLEMS=""

add_problem() {
  PROBLEMS+="${PROBLEMS:+; }$1"
}

# has_line LIST LINE: succeeds when LINE is one of the newline separated names in LIST.
has_line() {
  case $'\n'"$1"$'\n' in
    *$'\n'"$2"$'\n'*) return 0 ;;
  esac
  return 1
}

# check_container NAME: the container exists, sets no variable name beyond its image's own names and the
# template's, and mounts only volumes that are on the template.
check_container() {
  local c=$1 own name type vol
  dk inspect --type container --format "$F_IMAGE" "$c"
  if [ "$DK_RC" -ne 0 ] || [ -z "$DK_OUT" ]; then
    add_problem "container $c is missing"
    return 0
  fi
  dk inspect --type image --format "$F_ENV_NAMES" "$DK_OUT"
  if [ "$DK_RC" -ne 0 ]; then
    add_problem "the image of $c cannot be read"
    return 0
  fi
  own=$DK_OUT
  dk inspect --type container --format "$F_ENV_NAMES" "$c"
  if [ "$DK_RC" -ne 0 ]; then
    add_problem "the environment of $c cannot be read"
    return 0
  fi
  while IFS= read -r name; do
    [ -n "$name" ] || continue
    has_line "$own" "$name" && continue
    listed "env $c" "$name" || add_problem "container $c sets the name $name, which is not on the template"
  done <<<"$DK_OUT"
  dk inspect --type container --format "$F_MOUNT_TYPES" "$c"
  if [ "$DK_RC" -ne 0 ]; then
    add_problem "the mounts of $c cannot be read"
    return 0
  fi
  while read -r type vol; do
    [ -n "$type" ] || continue
    if [ "$type" != volume ]; then
      add_problem "container $c has a $type mount"
    elif ! listed volumes "$vol"; then
      add_problem "container $c mounts the volume $vol, which is not on the template"
    fi
  done <<<"$DK_OUT"
}

# Runs in the gateway container. Prints "<profile> <NAME>" for each line of each settings file in the
# platform home, and "<profile> ?" for a line that is not NAME=value. A value is never printed.
ENV_NAMES_SH='
[ -n "${HERMES_HOME-}" ] || exit 3
for f in "$HERMES_HOME/.env" "$HERMES_HOME"/profiles/*/.env; do
  [ -f "$f" ] || continue
  [ -r "$f" ] || exit 4
  if [ "$f" = "$HERMES_HOME/.env" ]; then w=default; else w=${f%/.env}; w=${w##*/}; fi
  case "$w" in *[!A-Za-z0-9._-]*) w="?" ;; esac
  while IFS= read -r l || [ -n "$l" ]; do
    l=${l#"${l%%[![:space:]]*}"}
    case "$l" in "" | "#"*) continue ;; esac
    n=${l%%=*}
    case "$l" in *=*) ;; *) n="?" ;; esac
    case "$n" in "" | [0-9]* | *[!A-Za-z0-9_]*) n="?" ;; esac
    printf "%s %s\n" "$w" "$n"
  done < "$f"
done
'

check_home_env_names() {
  local where name
  dk exec "$GATE_C" sh -c "$ENV_NAMES_SH"
  if [ "$DK_RC" -ne 0 ]; then
    add_problem "the settings files in the platform home cannot be read"
    return 0
  fi
  while read -r where name; do
    [ -n "$where" ] || continue
    if [ "$name" = "?" ]; then
      add_problem "the settings file of profile $where has a line without a name"
    elif ! listed home-env "$name"; then
      add_problem "the settings file of profile $where has the name $name, which is not on the template"
    fi
  done <<<"$DK_OUT"
}

# verdict CHECK: the one line of a check that collected its problems in PROBLEMS.
verdict() {
  if [ -z "$PROBLEMS" ]; then
    pass "$1"
  else
    fail "$1" "$PROBLEMS"
  fi
}

check_names_on_template() {
  local c
  PROBLEMS=""
  for c in "${CONTAINERS[@]}"; do
    check_container "$c"
  done
  check_home_env_names
  verdict names-on-template
}

# The platform home's two profiles: the default one and the planner.
PROFILES=(default raphael-planner)

# Runs in the gateway container: $1 names the profile, whose folder becomes the platform home.
PROFILE_SH='
[ -n "${HERMES_HOME-}" ] || exit 3
case "$1" in default) ;; *) HERMES_HOME=$HERMES_HOME/profiles/$1; export HERMES_HOME ;; esac
shift
'

# Runs the program on stdin with the interpreter that the hermes command names: the first hermes on PATH
# whose first line is a shebang for something other than a shell (the published image puts a shell shim first).
FIND_PY_SH='
p=$PATH:
found=
while [ -n "$p" ]; do
  d=${p%%:*}
  p=${p#*:}
  f=${d:-.}/hermes
  [ -f "$f" ] && [ -x "$f" ] || continue
  first=
  IFS= read -r first < "$f" || true
  case "$first" in "#!"*) ;; *) continue ;; esac
  set -f
  set -- ${first#"#!"}
  set +f
  case "${1##*/}" in sh | bash | dash | ash | ksh | zsh) continue ;; esac
  found=1
  break
done
[ -n "$found" ] || exit 127
exec "$@" -
'

PICKED=""

# pick PREFIX: the last line of DK_OUT that starts with PREFIX (empty when there is none).
pick() {
  local line
  PICKED=""
  while IFS= read -r line; do
    case "$line" in "$1"*) PICKED=$line ;; esac
  done <<<"$DK_OUT"
}

# profile_read PROFILE PROGRAM: runs a Python program (on stdin) in the gateway container, with that
# profile's folder as the platform home.
profile_read() {
  dk exec -i "$GATE_C" sh -c "$PROFILE_SH$FIND_PY_SH" sh "$1" <<<"$2"
}

DISPATCH_PY='from hermes_cli.config import load_config
kanban = load_config().get("kanban") or {}
print("dispatcher", "off" if kanban.get("dispatch_in_gateway") is False else "on")
'

check_dispatcher_off() {
  local p
  PROBLEMS=""
  for p in "${PROFILES[@]}"; do
    profile_read "$p" "$DISPATCH_PY"
    pick "dispatcher "
    if [ "$DK_RC" -ne 0 ] || [ -z "$PICKED" ]; then
      add_problem "the dispatcher setting of profile $p cannot be read"
    elif [ "$PICKED" != "dispatcher off" ]; then
      add_problem "the dispatcher of profile $p is on or unset"
    fi
  done
  verdict dispatcher-off
}

EMPTY_CRON=$'No scheduled jobs.\nCreate one with \'hermes cron create ...\' or the /cron command in chat.'

check_cron_lists_empty() {
  local p
  PROBLEMS=""
  for p in "${PROFILES[@]}"; do
    dk exec "$GATE_C" sh -c "${PROFILE_SH}exec hermes cron list" sh "$p"
    if [ "$DK_RC" -ne 0 ]; then
      add_problem "the cron job list of profile $p cannot be read"
    elif [ "$DK_OUT" != "$EMPTY_CRON" ]; then
      add_problem "profile $p has a cron job"
    fi
  done
  verdict cron-lists-empty
}

PLATFORM_PY='from hermes_cli.send_cmd import _load_hermes_env
_load_hermes_env()
from gateway.config import load_gateway_config
print("platforms", *sorted(p.value for p in load_gateway_config().get_connected_platforms()))
'

# Only the API server may be connected; any messaging platform fails the check.
check_no_messaging_platform() {
  local p n
  local -a names
  PROBLEMS=""
  for p in "${PROFILES[@]}"; do
    profile_read "$p" "$PLATFORM_PY"
    pick "platforms"
    if [ "$DK_RC" -ne 0 ] || [ -z "$PICKED" ]; then
      add_problem "the connected platforms of profile $p cannot be read"
      continue
    fi
    read -ra names <<<"${PICKED#platforms}"
    for n in "${names[@]}"; do
      [ "$n" = api_server ] || add_problem "profile $p connects the platform $n"
    done
  done
  verdict no-messaging-platform
}

SEND_TEXT='isolation check: this message must not be delivered'

# hermes send exits 1 when delivery fails (2 is a usage error, 0 a delivery). The shell in the container
# reports that status itself, so that a docker error or a missing command cannot pass as a failed delivery.
check_telegram_send_fails() {
  local p
  PROBLEMS=""
  for p in "${PROFILES[@]}"; do
    dk exec "$GATE_C" sh -c "${PROFILE_SH}"'hermes send --to telegram "$1" >/dev/null 2>&1; echo "send status $?"' \
      sh "$p" "$SEND_TEXT"
    pick "send status "
    if [ "$DK_RC" -ne 0 ] || [ -z "$PICKED" ]; then
      add_problem "the send to telegram from profile $p could not be tried"
    elif [ "$PICKED" = "send status 0" ]; then
      add_problem "the send to telegram from profile $p was delivered"
    elif [ "$PICKED" != "send status 1" ]; then
      add_problem "the send to telegram from profile $p did not fail with a delivery error (${PICKED#send })"
    fi
  done
  verdict telegram-send-fails
}

PROXY_SOURCE="$HERE/synthetic_proxy.py"

# Runs in the proxy container with the proxy's source on stdin. It loads the proxy as a module that is not
# main, reads the signing key that is already in the proxy's state folder and never makes one, and prints
# one assertion from the proxy's own mint_assertion.
MINT_PY='import sys, types
from pathlib import Path
from cryptography.hazmat.primitives import serialization
proxy = types.ModuleType("synthetic_proxy")
exec(compile(sys.stdin.read(), "synthetic_proxy.py", "exec"), proxy.__dict__)
pem = (Path(proxy.STATE) / "signing-key.pem").read_bytes()
proxy.SIGNING_KEY = serialization.load_pem_private_key(pem, password=None)
print(proxy.mint_assertion())
'

ORIGIN_RE='^https://[A-Za-z0-9]([A-Za-z0-9.-]*[A-Za-z0-9])?(:[0-9]{1,5})?/?$'
JWT_RE='^[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+$'

HOST=""

# host_of URL: sets HOST to the lower case host of an absolute URL, without user, port or trailing dot.
# HOST is empty when the text has no scheme and host.
host_of() {
  local auth
  HOST=""
  [[ $1 =~ ^[A-Za-z][A-Za-z0-9+.-]*://([^/?#]*) ]] || return 0
  auth=${BASH_REMATCH[1]##*@}
  case $auth in
    \[*\]*) auth=${auth%%]*}] ;;
    *) auth=${auth%%:*} ;;
  esac
  auth=${auth,,}
  while [[ $auth == *. ]]; do
    auth=${auth%.}
  done
  HOST=$auth
}

# probe_production ORIGIN: one GET with one proxy assertion. The assertion and the address travel to curl
# as its configuration on stdin, so neither is on a command line. Refused means 401, 403, or a redirect
# to another host; curl is told not to follow it.
probe_production() {
  local origin=$1 jwt origin_host code redirect rc=0 out
  if [ ! -r "$PROXY_SOURCE" ]; then
    add_problem "the proxy source is not readable"
    return 0
  fi
  dk exec -i "$PROXY_C" python3 -c "$MINT_PY" <"$PROXY_SOURCE"
  jwt=${DK_OUT##*$'\n'}
  if [ "$DK_RC" -ne 0 ] || ! [[ $jwt =~ $JWT_RE ]]; then
    add_problem "no assertion could be minted in the proxy container"
    return 0
  fi
  host_of "$origin"
  origin_host=$HOST
  out=$(printf 'url = "%s"\nheader = "Cf-Access-Jwt-Assertion: %s"\n' "$origin" "$jwt" |
    curl -K - --max-redirs 0 -o /dev/null --max-time 20 -w '%{http_code} %{redirect_url}' 2>/dev/null) || rc=$?
  if [ "$rc" -ne 0 ]; then
    add_problem "the request to production failed (curl status $rc)"
    return 0
  fi
  read -r code redirect <<<"$out"
  [[ $code =~ ^[0-9]{3}$ ]] || code=""
  case "$code" in
    401 | 403) ;;
    3[0-9][0-9])
      host_of "$redirect"
      if [ -z "$HOST" ]; then
        add_problem "production redirected, but not to a host (status $code)"
      elif [ "$HOST" = "$origin_host" ]; then
        add_problem "production redirected to its own host (status $code)"
      fi
      ;;
    *) add_problem "production answered with status ${code:-none}, not a refusal" ;;
  esac
}

check_production_get_refused() {
  local origin=${JOURNEY_PRODUCTION_ORIGIN-}
  PROBLEMS=""
  if [ -z "$origin" ]; then
    add_problem "JOURNEY_PRODUCTION_ORIGIN is not set"
  elif ! [[ $origin =~ $ORIGIN_RE ]]; then
    add_problem "JOURNEY_PRODUCTION_ORIGIN is not a plain https address"
  else
    probe_production "$origin"
  fi
  verdict production-get-refused
}

run_server2() {
  require_stack_host
  check_names_on_template
  check_dispatcher_off
  check_cron_lists_empty
  check_no_messaging_platform
  check_telegram_send_fails
  check_production_get_refused
}

# normalized TEXT: sets NORMAL to TEXT in lower case, without blanks and trailing slashes.
NORMAL=""
normalized() {
  local text=${1,,}
  text=${text//[[:space:]]/}
  while [[ $text == */ ]]; do
    text=${text%/}
  done
  NORMAL=$text
}

# Server 1: production's owner app must name an issuer that is not the proxy's. The one variable is read
# by name inside the production container and compared in this shell; neither value is printed or passed
# on a command line.
check_production_issuer_differs() {
  local line proxy_issuer="" production_issuer
  PROBLEMS=""
  if [ ! -r "$PROXY_SOURCE" ]; then
    add_problem "the proxy source is not readable"
    verdict production-issuer-differs
    return 0
  fi
  while IFS= read -r line; do
    if [[ $line =~ ^ISSUER\ =\ \"([^\"]*)\"$ ]]; then
      proxy_issuer=${BASH_REMATCH[1]}
      break
    fi
  done <"$PROXY_SOURCE"
  normalized "$proxy_issuer"
  proxy_issuer=$NORMAL
  if [ -z "$proxy_issuer" ]; then
    add_problem "the proxy's issuer cannot be read from its source"
    verdict production-issuer-differs
    return 0
  fi
  dk exec "$PROD_C" printenv "$PROD_VAR"
  normalized "$DK_OUT"
  production_issuer=$NORMAL
  DK_OUT=""
  if [ "$DK_RC" -ne 0 ]; then
    add_problem "$PROD_VAR cannot be read in the production container"
  elif [ -z "$production_issuer" ]; then
    add_problem "production sets no value for $PROD_VAR"
  elif [ "$production_issuer" = "$proxy_issuer" ]; then
    add_problem "production names the proxy's issuer in $PROD_VAR"
  fi
  verdict production-issuer-differs
}

run_server1() {
  check_production_issuer_differs
}

case "$MODE" in
  server2) run_server2 ;;
  server1) run_server1 ;;
esac

exit "$FAILED"
