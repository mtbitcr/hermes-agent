#!/usr/bin/env bash
# Reset the journeys test stack on Server 2 to a fresh, test-data-only state (card P-2).
# Every container, volume and variable name is read from stack.env.template at run time.
# No access value is put on a command line or printed; values travel on standard input only.
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
template=$here/stack.env.template
stack_home=$here/stack-home

changed=0
say() { printf 'reset_stack: %s\n' "$*"; }
die() {
  printf 'reset_stack: %s\n' "$*" >&2
  if ((changed)); then
    # The guards refuse a stopped dashboard, so a half-done reset needs this hint.
    printf 'reset_stack: the reset stopped part-way; start the dashboard container if it is down, then run it again\n' >&2
  fi
  exit 1
}

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

# The Go templates below, and fmt_app after the variable lists, are the only docker --format shapes this script uses.
fmt_image='{{.Image}}'
fmt_dest='{{range .Mounts}}{{println .Name .Destination}}{{end}}'
fmt_state='{{.State.Running}} {{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}'
fmt_home='{{range .Config.Env}}{{$p := split . "="}}{{if eq (index $p 0) "HERMES_HOME"}}{{index $p 1}}{{end}}{{end}}'
fmt_hc='{{if .Config.Healthcheck}}{{index .Config.Healthcheck.Test 0}}{{else}}NONE{{end}}'

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
proxy_c=${containers[0]} app_c=${containers[1]} dash_c=${containers[2]}
gate_c=${containers[3]} browser_c=${containers[4]}
home_v=${volumes[0]} tokens_v=${volumes[1]}

surfaces=(workspace recommendations connections automations models)
surface_env=(HERMES_API_TOKEN_FILE HERMES_RECOMMENDATIONS_TOKEN_FILE HERMES_CONNECTIONS_API_TOKEN_FILE
  HERMES_AUTOMATIONS_API_TOKEN_FILE HERMES_MODELS_API_TOKEN_FILE)
executor_env=HERMES_EXECUTOR_API_TOKEN_FILE
planner_env=HERMES_PLANNER_API_TOKEN_FILE

# Owner app settings, read only while the owner app is stopped or has never run: its user, then each token file
# setting as one %q line, so no value spans lines or reads as plain.
fmt_app='{{println (printf "%q" .Config.User)}}{{range .Config.Env}}{{$p := split . "="}}{{if eq (index $p 0)'
for name in "${surface_env[@]}" "$executor_env" "$planner_env"; do fmt_app+=" \"$name\""; done
fmt_app+='}}{{println (printf "%q" .)}}{{end}}{{end}}'

for name in JOURNEY_STACK_HOST JOURNEY_MODEL_TOKEN_FILE; do
  is_listed script-input "$name" || die "$name is not on the template ([script-input])"
done
api_key_var=API_SERVER_KEY
model_var=CLAUDE_CODE_OAUTH_TOKEN
for name in "$api_key_var" "$model_var"; do
  is_listed home-env "$name" || die "$name is not on the template ([home-env])"
done
for name in "${surface_env[@]}" "$executor_env" "$planner_env"; do
  is_listed "env $app_c" "$name" || die "$name is not on the template for the owner app container"
done
for c in "$dash_c" "$gate_c"; do
  is_listed "env $c" HERMES_HOME || die "HERMES_HOME is not on the template for $c"
done

# --- Guards: nothing is changed before all of them pass.
[[ -n ${JOURNEY_STACK_HOST:-} ]] || die "JOURNEY_STACK_HOST is not set; nothing was changed"
[[ $(hostname) == "$JOURNEY_STACK_HOST" ]] ||
  die "the host name does not equal JOURNEY_STACK_HOST; nothing was changed"

missing=()
for c in "${containers[@]}"; do
  if found_image=$(inspect "$fmt_image" "$c" 2> /dev/null); then
    [[ $c != "$app_c" ]] || app_image=$found_image
  else
    missing+=("$c")
  fi
done
((${#missing[@]} == 0)) || die "container not found: ${missing[*]}; nothing was changed"

# A container answers when its own Docker health check says healthy, so every container must have one. The
# shipped platform image declares none, and the facts give no port or probe command for the proxy, the
# gateway API or the browser, so this script must not invent a probe: the operator's health check is the
# contract. A container without a configured check (kind NONE, or one that cannot be read) is refused here.
for c in "${containers[@]}"; do
  kind=$(inspect "$fmt_hc" "$c" 2> /dev/null) || kind=""
  [[ $kind == CMD || $kind == CMD-SHELL ]] ||
    die "$c has no configured Docker health check; the operator must create it with a health check; nothing was changed"
done

# Only the dashboard must run, as the platform's account is read inside it. The owner app is restarted below, so
# it may be stopped or have never run: its state only decides how its numbers and token file names are read.
app_state=$(inspect "$fmt_state" "$app_c") || die "cannot read the state of $app_c; nothing was changed"
state=$(inspect "$fmt_state" "$dash_c") || die "cannot read the state of $dash_c; nothing was changed"
[[ $state == "true "* ]] || die "$dash_c is not running; nothing was changed"
app_running=0
if [[ $app_state == "true "* ]]; then app_running=1; fi

app_tokens=$(mount_dest "$app_c" "$tokens_v") ||
  die "$app_c must mount the token volume exactly once; nothing was changed"
dash_tokens=$(mount_dest "$dash_c" "$tokens_v") ||
  die "$dash_c must mount the token volume exactly once; nothing was changed"
dash_home=$(mount_dest "$dash_c" "$home_v") ||
  die "$dash_c must mount the home volume exactly once; nothing was changed"
gate_home=$(mount_dest "$gate_c" "$home_v") ||
  die "$gate_c must mount the home volume exactly once; nothing was changed"
# The configured HERMES_HOME of each container must be where it mounts the home volume, or the reset
# would empty one directory while the container keeps its state in another. Read for a stopped one too.
for pair in "$dash_c|$dash_home" "$gate_c|$gate_home"; do
  hermes_home=$(inspect "$fmt_home" "${pair%%|*}") ||
    die "cannot read HERMES_HOME of ${pair%%|*}; nothing was changed"
  [[ $hermes_home == "${pair#*|}" ]] ||
    die "HERMES_HOME of ${pair%%|*} is not the home volume mount; nothing was changed"
done

# read_number TEXT: TEXT as one positive whole number.
read_number() {
  [[ $1 =~ ^[1-9][0-9]{0,9}$ ]] || return 1
  printf '%s' "$1"
}

# Two identities that must never be mixed: the owner app's user, and the platform's runtime account,
# which is whoever owns the top folder of the home inside the running dashboard (the boot hook sets it).
# A running owner app tells its own numbers by docker exec. One that is stopped or has never run cannot, so its
# numbers come from its settings and one throw-away run of its image.
if ((app_running)); then
  app_uid=$(docker exec "$app_c" id -u 2> /dev/null) || app_uid=""
  app_uid=$(read_number "$app_uid") ||
    die "the user id of $app_c is not a positive number; nothing was changed"
  app_gid=$(docker exec "$app_c" id -g 2> /dev/null) || app_gid=""
  app_gid=$(read_number "$app_gid") ||
    die "the group id of $app_c is not a positive number; nothing was changed"
else
  # The owner app's settings stay in this shell: nothing prints them or puts one on a command line.
  app_settings=$(inspect "$fmt_app" "$app_c" 2> /dev/null) ||
    die "cannot read the settings of $app_c; nothing was changed"
  mapfile -t app_lines <<< "$app_settings"

  # Its numbers: one throw-away run of its image as its configured user (the one setting on a command line).
  quoted_re='^"([^"\\]*)"$' user_re='^[A-Za-z0-9_][A-Za-z0-9_.-]*(:[A-Za-z0-9_][A-Za-z0-9_.-]*)?$'
  app_user=""
  if [[ ${app_lines[0]-} =~ $quoted_re ]]; then app_user=${BASH_REMATCH[1]}; fi
  [[ $app_user =~ $user_re ]] ||
    die "the user id of $app_c is not a positive number; nothing was changed"
  # Fail closed: the closing x keeps the exact output, because $(...) would drop trailing blank lines.
  probe_raw=$(docker run --rm --network none --user "$app_user" --entrypoint sh "$app_image" \
    -c 'id -u && id -g' 2> /dev/null && printf x) ||
    die "the identity probe of $app_c did not exit 0; nothing was changed"
  probe_out=${probe_raw%x}
  mapfile -t probe_ids < <(printf '%s' "$probe_out")
  [[ $probe_out == *$'\n' ]] && ((${#probe_ids[@]} == 2)) ||
    die "the identity probe of $app_c did not print exactly two lines; nothing was changed"
  app_uid=$(read_number "${probe_ids[0]}") ||
    die "the user id of $app_c is not a positive number; nothing was changed"
  app_gid=$(read_number "${probe_ids[1]}") ||
    die "the group id of $app_c is not a positive number; nothing was changed"
fi
platform_owner=$(docker exec "$dash_c" stat -c %u:%g "$dash_home" 2> /dev/null) || platform_owner=""
[[ $platform_owner =~ ^[1-9][0-9]{0,9}:[1-9][0-9]{0,9}$ ]] ||
  die "the owner of the home folder in $dash_c is not two positive numbers; nothing was changed"
platform_uid=${platform_owner%%:*} platform_gid=${platform_owner##*:}
app_id=$app_uid:$app_gid platform_id=$platform_uid:$platform_gid

# token_file_name VARIABLE: the plain file name that the owner app has for VARIABLE, directly under its token mount.
# A running owner app prints it by docker exec; for one that is stopped or has never run, its settings give it once.
token_file_name() {
  local line value="" found=0 start='"'"$1="
  if ((app_running)); then
    value=$(docker exec "$app_c" printenv "$1" 2> /dev/null) || return 1
  else
    for line in "${app_lines[@]:1}"; do
      if [[ $line == "$start"* ]]; then
        value=${line#"$start"}
        value=${value%'"'}
        found=$((found + 1))
      fi
    done
    ((found == 1)) || return 1
  fi
  [[ $value == "$app_tokens"/* ]] || return 1
  value=${value#"$app_tokens"/}
  [[ $value =~ ^[A-Za-z0-9_][A-Za-z0-9_.-]*$ ]] || return 1
  printf '%s' "$value"
}

declare -A token_file=()
seen_files=" "
for name in "${surface_env[@]}" "$executor_env" "$planner_env"; do
  file=$(token_file_name "$name") ||
    die "$name in $app_c is not a plain file name directly under the token volume mount; nothing was changed"
  [[ $seen_files != *" $file "* ]] ||
    die "$name in $app_c shares its token file with another variable; nothing was changed"
  seen_files+="$file "
  token_file[$name]=$file
done

# The optional model token is read once into memory: never exported, never on a command line.
model_token=""
token_source=${JOURNEY_MODEL_TOKEN_FILE:-}
if [[ -n $token_source && -e $token_source ]]; then
  [[ -f $token_source && -r $token_source ]] ||
    die "JOURNEY_MODEL_TOKEN_FILE is not a readable file; nothing was changed"
  model_token=$(< "$token_source")
  model_token=${model_token#"${model_token%%[![:space:]]*}"}
  model_token=${model_token%"${model_token##*[![:space:]]}"}
  [[ -z $model_token || $model_token =~ ^[[:graph:]]+$ ]] ||
    die "JOURNEY_MODEL_TOKEN_FILE must hold a single word; nothing was changed"
fi
if [[ -z $model_token ]]; then
  say "model token step skipped: JOURNEY_MODEL_TOKEN_FILE is unset, missing or empty"
fi

planner_profile=raphael-planner
for settings in env.settings "profiles/$planner_profile/env.settings"; do
  [[ -f $stack_home/$settings ]] || die "stack-home lacks $settings; nothing was changed"
done
image=$(inspect "$fmt_image" "$dash_c") ||
  die "cannot read the image of $dash_c; nothing was changed"

# --- Stop the gateway and the dashboard, then empty the home and token volumes.
# run_helper VOLUME SCRIPT [-i]: a throw-away root container that sees only that volume.
run_helper() {
  local volume=$1 script=$2
  shift 2
  docker run --rm "$@" --network none --user 0 --entrypoint sh \
    --mount "type=volume,source=$volume,target=/volume" "$image" -c "$script"
}

home_script=$(
  cat <<'EOF'
set -e
find /volume -mindepth 1 -delete
tar -xf - -C /volume
find /volume -name env.settings -exec sh -c 'set -e; for f; do grep -v "^[[:space:]]*#" "$f" > "${f%env.settings}.env" || [ $? -eq 1 ]; chmod 600 "${f%env.settings}.env"; rm "$f"; done' sh {} +
chown -R @PLATFORM@ /volume
chmod go-rwx /volume
EOF
)
tokens_script=$(
  cat <<'EOF'
set -e
find /volume -mindepth 1 -delete
chown @PLATFORM@ /volume
chmod 700 /volume
EOF
)
handover_script=$(
  cat <<'EOF'
set -e
chown -R @APP@ /volume
EOF
)
# The ids were checked above to be plain numbers.
home_script=${home_script//@PLATFORM@/$platform_id}
tokens_script=${tokens_script//@PLATFORM@/$platform_id}
handover_script=${handover_script//@APP@/$app_id}

# The owner app reads its Project, Connections and Automations through a Project of this slug and a board of the same
# slug whose metadata names that Project's id. The program runs as the platform's runtime account, on the interpreter
# that the first real hermes launcher on PATH names (the published image puts a shell shim first).
anchor=raphael-workspace
anchor_program=$(
  cat <<'EOF'
import sys
from hermes_cli import kanban_db, projects_db

slug = "@ANCHOR@"
with projects_db.connect_closing() as conn:
    project = projects_db.get_project(conn, slug)
if project is None or project.slug != slug:
    sys.exit("the Project is not in the registry")
kanban_db.write_board_metadata(slug, project_id=project.id)
meta = kanban_db.read_board_metadata(slug)
if not kanban_db.board_exists(slug) or meta.get("slug") != slug or meta.get("project_id") != project.id:
    sys.exit("the board metadata does not name the Project id")
EOF
)
anchor_program=${anchor_program//@ANCHOR@/$anchor}
platform_python=$(
  cat <<'EOF'
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
EOF
)

changed=1
for c in "$gate_c" "$dash_c"; do
  say "stopping $c"
  docker stop "$c" > /dev/null || die "cannot stop $c"
done
say "emptying the token volume $tokens_v"
run_helper "$tokens_v" "$tokens_script" > /dev/null || die "cannot empty the token volume $tokens_v"
say "emptying the home volume $home_v and writing the home from stack-home"
tar -C "$stack_home" -cf - . | run_helper "$home_v" "$home_script" -i > /dev/null ||
  die "cannot write the home volume $home_v"

# --- Make the Project, its board and the stack-only access values inside the dashboard container, as the platform's
# runtime account. The Project and its board come first: no token is issued for an app that could not read its Project.
say "starting $dash_c"
docker start "$dash_c" > /dev/null || die "cannot start $dash_c"

say "creating the Project $anchor and its board"
docker exec --user "$platform_id" "$dash_c" hermes project create "Raphael Workspace" \
  --slug "$anchor" --board "$anchor" > /dev/null 2>&1 ||
  die "cannot create the Project $anchor and its board"
say "binding the board $anchor to the id of the Project $anchor"
docker exec -i --user "$platform_id" "$dash_c" sh -c "$platform_python" sh <<< "$anchor_program" > /dev/null 2>&1 ||
  die "cannot bind the board $anchor to the id of the Project $anchor, or the id did not read back"

for i in "${!surfaces[@]}"; do
  say "issuing the ${surfaces[i]} token"
  docker exec --user "$platform_id" "$dash_c" hermes kanban-workspace-token issue \
    --surface "${surfaces[i]}" --out "$dash_tokens/${token_file[${surface_env[i]}]}" > /dev/null 2>&1 ||
    die "cannot issue the ${surfaces[i]} token"
done

# new_api_key ENV_FILE TOKEN_FILE: the key is made inside the container, appended to the profile's
# .env and written to the owner app's token file there; it never reaches this script.
new_api_key() {
  docker exec --user "$platform_id" "$dash_c" sh -c 'set -e
umask 077
key=$(head -c 32 /dev/urandom | od -An -tx1 | tr -d " \n")
test "${#key}" -eq 64
printf "%s=%s\n" "$3" "$key" >> "$1"
printf "%s\n" "$key" > "$2"' sh "$1" "$2" "$api_key_var" > /dev/null 2>&1
}

# add_model_token ENV_FILE: the value goes in on standard input only.
add_model_token() {
  printf '%s=%s\n' "$model_var" "$model_token" |
    docker exec -i --user "$platform_id" "$dash_c" sh -c 'cat >> "$1"' sh "$1" > /dev/null 2>&1
}

profile_label=(default "$planner_profile")
profile_env=("$dash_home/.env" "$dash_home/profiles/$planner_profile/.env")
key_file=("${token_file[$executor_env]}" "${token_file[$planner_env]}")
for i in 0 1; do
  say "making the API server key of the ${profile_label[i]} profile"
  new_api_key "${profile_env[i]}" "$dash_tokens/${key_file[i]}" ||
    die "cannot make the API server key of the ${profile_label[i]} profile"
done
if [[ -n $model_token ]]; then
  for i in 0 1; do
    say "adding the model token to the ${profile_label[i]} profile"
    add_model_token "${profile_env[i]}" ||
      die "cannot add the model token to the ${profile_label[i]} profile"
  done
fi

# --- The owner app reads the token files as its own user; the platform account needs them no more.
say "handing the token volume $tokens_v to the owner app"
run_helper "$tokens_v" "$handover_script" > /dev/null || die "cannot hand the token volume $tokens_v to the owner app"

# --- Restart the owner app (it keeps token values in memory) and the dashboard, start the rest.
say "restarting $app_c so that it reads the new token files"
docker restart "$app_c" > /dev/null || die "cannot restart $app_c"
say "restarting $dash_c"
docker restart "$dash_c" > /dev/null || die "cannot restart $dash_c"
for c in "$gate_c" "$proxy_c" "$browser_c"; do
  say "starting $c"
  docker start "$c" > /dev/null || die "cannot start $c"
done

# answers CONTAINER: it runs and its health check reports healthy; starting, unhealthy and no status are not yet.
answers() {
  local state
  state=$(inspect "$fmt_state" "$1" 2> /dev/null) || return 1
  [[ $state == "true healthy" ]]
}

# One budget of 120 seconds for all containers, polled every 2 seconds.
changed=0
waited=0
pending=("$app_c" "$dash_c" "$gate_c" "$proxy_c" "$browser_c")
while :; do
  still=()
  for c in "${pending[@]}"; do
    if ! answers "$c"; then still+=("$c"); fi
  done
  pending=("${still[@]}")
  if ((${#pending[@]} == 0)); then break; fi
  if ((waited >= 120)); then
    die "no answer after 120 seconds from: ${pending[*]}"
  fi
  sleep 2
  waited=$((waited + 2))
done
say "the stack is reset and every container answers"
