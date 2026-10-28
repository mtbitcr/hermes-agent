#!/usr/bin/env bash
# Seed the journeys test stack on Server 2 for one journey (card 2): the project, and the waiting owner question
# and suggestion (journey 3) or the card stopped after repeated failures (journey 5). Run it after reset_stack.sh.
# The names come from seed.json of the released owner app copy in the browser container; seed_states.py makes
# the states inside the dashboard container, as the platform's runtime account and never as root.
# Every container and volume name is read from stack.env.template at run time. No access value is put on a
# command line or printed; seed.json travels on standard input.
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
template=$here/stack.env.template
program=$here/seed_states.py
seed_file=/opt/raphael-journeys/tools/page-check/journeys/seed.json

say() { printf 'seed_stack: %s\n' "$*"; }
die() {
  printf 'seed_stack: %s\n' "$*" >&2
  exit 1
}

(($# == 1)) && [[ $1 =~ ^[0-9]+$ ]] || die "usage: seed_stack.sh JOURNEY (the journey number)"
journey=$1
[[ -f $program ]] || die "seed_states.py is missing"

declare -A listed=()

read_template() {
  local line section="" section_re='^\[([^]]+)\]$'
  [[ -f $template ]] || die "stack.env.template is missing"
  while IFS= read -r line || [[ -n $line ]]; do
    line=${line#"${line%%[![:space:]]*}"}
    line=${line%"${line##*[![:space:]]}"}
    if [[ -z $line || $line == "#"* ]]; then
      continue
    fi
    if [[ $line =~ $section_re ]]; then
      section=${BASH_REMATCH[1]}
    elif [[ -z $section ]]; then
      die "stack.env.template has a name outside a section"
    elif [[ $line =~ ^[A-Za-z0-9_.-]+$ ]]; then
      listed["$section"]+="$line"$'\n'
    else
      die "stack.env.template holds more than a name in [$section]"
    fi
  done < "$template"
}

names_in() { printf '%s' "${listed["$1"]-}"; }

is_listed() {
  local name
  while IFS= read -r name; do
    if [[ $name == "$2" ]]; then return 0; fi
  done < <(names_in "$1")
  return 1
}

# The Go templates below are the only docker --format shapes this script uses.
fmt_dest='{{range .Mounts}}{{println .Name .Destination}}{{end}}'
fmt_state='{{.State.Running}} {{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}'
fmt_home='{{range .Config.Env}}{{$p := split . "="}}{{if eq (index $p 0) "HERMES_HOME"}}{{index $p 1}}{{end}}{{end}}'

inspect() { docker inspect --type container --format "$1" "$2"; }

# mount_dest CONTAINER VOLUME: the one place the container mounts the volume.
mount_dest() {
  local out name dest found=()
  out=$(inspect "$fmt_dest" "$1") || return 1
  while read -r name dest; do
    if [[ $name == "$2" ]]; then found+=("$dest"); fi
  done <<< "$out"
  ((${#found[@]} == 1)) || return 1
  printf '%s' "${found[0]}"
}

read_template
mapfile -t containers < <(names_in containers)
mapfile -t volumes < <(names_in volumes)
if ((${#containers[@]} != 5 || ${#volumes[@]} != 3)); then
  die "the template must list five containers and three volumes"
fi
# Roles are the order in the template: proxy, owner app, dashboard, gateway, browser.
app_c=${containers[1]} dash_c=${containers[2]} browser_c=${containers[4]}
home_v=${volumes[0]}

is_listed script-input JOURNEY_STACK_HOST || die "JOURNEY_STACK_HOST is not on the template ([script-input])"

# --- Guards: nothing is changed before all of them pass.
[[ -n ${JOURNEY_STACK_HOST:-} ]] || die "JOURNEY_STACK_HOST is not set; nothing was changed"
[[ $(hostname) == "$JOURNEY_STACK_HOST" ]] ||
  die "the host name does not equal JOURNEY_STACK_HOST; nothing was changed"

for c in "$app_c" "$dash_c" "$browser_c"; do
  state=$(inspect "$fmt_state" "$c") || die "cannot read the state of $c; nothing was changed"
  [[ $state == "true "* ]] || die "$c is not running; nothing was changed"
done

dash_home=$(mount_dest "$dash_c" "$home_v") ||
  die "$dash_c must mount the home volume exactly once; nothing was changed"
hermes_home=$(inspect "$fmt_home" "$dash_c") || die "cannot read HERMES_HOME of $dash_c; nothing was changed"
[[ $hermes_home == "$dash_home" ]] ||
  die "HERMES_HOME of $dash_c is not the home volume mount; nothing was changed"

# read_number CONTAINER COMMAND...: one positive whole number that the command prints inside the container.
read_number() {
  local out
  out=$(docker exec "$@" 2> /dev/null) || return 1
  [[ $out =~ ^[1-9][0-9]{0,9}$ ]] || return 1
  printf '%s' "$out"
}

# Two identities that must never be mixed: the owner app's user, and the platform's runtime account,
# which is whoever owns the top folder of the home inside the running dashboard.
app_uid=$(read_number "$app_c" id -u) ||
  die "the user id of $app_c is not a positive number; nothing was changed"
platform_owner=$(docker exec "$dash_c" stat -c %u:%g "$dash_home" 2> /dev/null) || platform_owner=""
[[ $platform_owner =~ ^[1-9][0-9]{0,9}:[1-9][0-9]{0,9}$ ]] ||
  die "the owner of the home folder in $dash_c is not two positive numbers; nothing was changed"
platform_uid=${platform_owner%%:*}
[[ $platform_uid != "$app_uid" ]] ||
  die "the home folder in $dash_c belongs to the owner app's user; nothing was changed"
platform_id=$platform_owner

seed_text=$(docker exec --user "$platform_id" "$browser_c" cat "$seed_file" 2> /dev/null) ||
  die "cannot read seed.json in $browser_c; nothing was changed"

# The platform's Python is named on the first line of the hermes launcher that the dashboard's PATH reaches.
python=$(docker exec --user "$platform_id" "$dash_c" sh -c 'IFS=:
for d in $PATH; do
  [ -f "$d/hermes" ] || continue
  IFS= read -r line < "$d/hermes" || true
  case $line in
    "#!/"*/python*)
      line=${line#??}
      printf "%s" "${line%% *}"
      exit 0
      ;;
  esac
done
exit 1' sh 2> /dev/null) || die "cannot find the platform's Python in $dash_c; nothing was changed"
[[ $python =~ ^/[A-Za-z0-9_./-]+$ ]] || die "the platform's Python in $dash_c is not an absolute path; nothing was changed"

say "seeding journey $journey in $dash_c"
printf '%s\n' "$seed_text" |
  docker exec -i --user "$platform_id" "$dash_c" "$python" -c "$(< "$program")" "$journey" ||
  die "the seed did not finish; if it wrote anything, run reset_stack.sh before seeding again"
say "journey $journey is seeded"
