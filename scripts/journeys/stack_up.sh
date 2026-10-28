#!/usr/bin/env bash
# Create the journeys test stack on Server 2 once, and bring it to the wanted versions on every run (card 2).
# Run by the runner identity with rootless Docker and no root. Every container, volume and variable name is
# read from stack.env.template at run time. No value is put on a command line or printed: the owner app
# settings reach docker through --env-file only.
#
# USAGE (for card 3, refresh_stack.sh on Server 1)
#   stack_up.sh versions
#   stack_up.sh build proxy
#   stack_up.sh build app <owner-app-revision>
#   stack_up.sh build browser <owner-app-revision>
#   stack_up.sh build platform <platform-commit> <archive-file>
#   stack_up.sh converge <proxy-tag> <app-revision> <platform-commit> <browser-revision>
#
# versions   One line per container, in the order of the template, as plain words: "<role> <tag>" for the roles
#            proxy, app, dashboard, gateway and browser. <tag> is the tag of the image that container was made
#            from, or "none" when the container does not exist. The app and browser tags are
#            "<revision>-<12 hex of the CA hash>"; the dashboard and gateway tags are the platform commit.
# build      Builds one image from scripts/journeys/images and prints one line "stack_up: built <role> <tag>".
#            proxy: tag = first 12 hex of sha256 over synthetic_proxy.py followed by images/proxy.Dockerfile.
#            app and browser: need the running proxy container; the CA is copied out as ca.pem and the tag is
#            "<revision>-<first 12 hex of sha256 of ca.pem>". The app image is raphael-journeys-app-base:<revision>
#            (loaded by the caller) plus the CA layer.
#            platform: starts the docker build of the archive (a tar of the commit) detached, with its own lock
#            and log under ${TMPDIR:-/tmp}, prints one line "stack_up: building platform <commit>; log <path>"
#            and exits 0 at once. When the image exists it prints "stack_up: image <ref> already exists"; when a
#            build holds the lock it prints "stack_up: a platform build is already running".
# converge   Creates what is missing and recreates what is stale. The proxy image
#            raphael-journeys-proxy:<proxy-tag>, the platform image raphael-journeys-platform:<platform-commit>
#            and raphael-journeys-app-base:<app-revision> must exist already; converge builds the app and
#            browser layers itself after the proxy has written its CA. Tags are plain: letters, digits, dot,
#            dash, underscore, at most 63.
#            Order: network, volumes, proxy, CA copy, CA image layers, dashboard, owner app, gateway, browser.
#            The owner app is on the stack network with the alias "workspace" (the proxy's upstream, port
#            3000). The gateway container runs the plain "gateway run" (the image hands it to its supervised
#            gateway service; HERMES_DASHBOARD stays unset) and shares the owner app's network namespace
#            (--network container:<owner app id>), so it has no alias or port of its own.
#            A new proxy tag recreates the proxy; a new owner app tag recreates the owner app and the gateway
#            (never a volume); a new browser tag the browser (card 3 passes the owner app revision for both, so
#            they change together); a new platform commit the dashboard and the gateway. The gateway is also
#            recreated when its network mode does not name the current owner app, or when it runs beside a
#            stopped owner app (the namespace it joined is gone). A container that is stopped but made by this
#            script is started when its turn comes.
#            Converge never issues a token. Before it starts or creates the owner app or the gateway it makes
#            one read-only check of the token volume: a throw-away helper container with no network mounts the
#            volume read-only and tests each of the token files that stack-app.settings names with test -s
#            (missing or zero size fails); it reads no content and prints nothing. A converge in which both
#            are running and unchanged makes no check.
#            Last line, exit 0, when every named token file exists: the owner app (the gateway after it) is
#            started if it is not running, all five containers are waited for (one budget of 180 seconds,
#            polled every 2 seconds) and the line is "stack_up: the stack answers".
#            Last line, exit 0, while a named token file is missing: the owner app and the gateway are only
#            created (never started, never waited for), the proxy, the dashboard and the browser are waited
#            for, and the line is "stack_up: the first reset starts the owner app and the gateway". Running
#            converge again changes nothing and ends with the same line until the files exist.
#
# Exit codes: 0 done; 1 refused or failed (the message is on standard error as "stack_up: ..."; a refusal made
# before any change ends with "nothing was changed"); 2 wrong usage. Run converge again after any failure.
#
# It refuses, before it changes anything, when: the host name is not JOURNEY_STACK_HOST; a tag is not plain;
# Docker is not rootless; less than 20 GB is free in Docker's data folder; a container with a template name
# exists without the label raphael-journeys.made-by=stack_up; or stack-app.settings names a variable that is not
# on the template for the owner app, or its issuer is not the proxy's own.
#
# Only containers with that label are ever changed, and only by container id. No volume is ever removed or
# written: this script has no volume remove, no prune and no docker rm -v, and its one helper mounts the token
# volume read-only. The proxy volume is mounted by the proxy only.
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
template=$here/stack.env.template
images=$here/images
settings=$here/stack-app.settings
proxy_source=$here/synthetic_proxy.py

label_value=stack_up
label=raphael-journeys.made-by=$label_value
network=raphael-journeys-net
token_dir=/var/lib/raphael-tokens
data_dir=/opt/data
state_dir=/state

say() { printf 'stack_up: %s\n' "$*"; }
die() {
  printf 'stack_up: %s\n' "$*" >&2
  exit 1
}
refuse() { die "$*; nothing was changed"; }
usage() {
  printf 'usage: stack_up.sh versions | build proxy | build app|browser <revision> | build platform <commit> <archive>\n' >&2
  printf '       stack_up.sh converge <proxy-tag> <app-revision> <platform-commit> <browser-revision>\n' >&2
  exit 2
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

read_template
mapfile -t containers < <(names_in containers)
mapfile -t volumes < <(names_in volumes)
if ((${#containers[@]} != 5 || ${#volumes[@]} != 3)); then
  die "the template must list five containers and three volumes"
fi
# Roles are the order in the template: proxy, owner app, dashboard, gateway, browser; home, tokens, proxy keys.
proxy_c=${containers[0]} app_c=${containers[1]} dash_c=${containers[2]}
gate_c=${containers[3]} browser_c=${containers[4]}
home_v=${volumes[0]} tokens_v=${volumes[1]} proxy_v=${volumes[2]}

# The only docker --format shapes this script uses.
fmt_name='{{.Name}}'
fmt_id='{{.Id}}'
fmt_ident='{{.Id}} {{index .Config.Labels "raphael-journeys.made-by"}}'
fmt_config_image='{{.Config.Image}}'
fmt_network_mode='{{.HostConfig.NetworkMode}}'
fmt_state='{{.State.Running}} {{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}'
fmt_security='{{range .SecurityOptions}}{{println .}}{{end}}'
fmt_root='{{.DockerRootDir}}'

# exists KIND FORMAT REF: read-only look at a network, volume or image.
exists() { docker inspect --type "$1" --format "$2" "$3" > /dev/null 2>&1; }

# ident NAME: sets found_id (empty when there is no such container) and found_label (read only).
found_id="" found_label=""
ident() {
  local out
  found_id="" found_label=""
  out=$(docker inspect --type container --format "$fmt_ident" "$1" 2> /dev/null) || return 0
  found_id=${out%% *}
  found_label=${out#* }
}

plain() { [[ $1 =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,62}$ ]]; }

# --- Guards: nothing is changed before all of them pass.
guard_host() {
  [[ -n ${JOURNEY_STACK_HOST:-} ]] || refuse "JOURNEY_STACK_HOST is not set"
  [[ $(hostname) == "$JOURNEY_STACK_HOST" ]] || refuse "the host name does not equal JOURNEY_STACK_HOST"
}

guard_docker() {
  local security root avail
  security=$(docker info --format "$fmt_security" 2> /dev/null) || refuse "docker does not answer"
  grep -qx 'name=rootless' <<< "$security" || refuse "the runner's Docker is not rootless"
  root=$(docker info --format "$fmt_root" 2> /dev/null) || root=""
  [[ -n $root ]] || refuse "cannot read Docker's data folder"
  avail=""
  read -r _ _ _ avail _ < <(df -Pk "$root" | tail -n 1) || true
  [[ $avail =~ ^[0-9]+$ ]] || refuse "cannot read the free space of Docker's data folder"
  ((avail >= 20 * 1024 * 1024)) || refuse "less than 20 GB is free in Docker's data folder"
}

guard_ownership() {
  local c
  for c in "${containers[@]}"; do
    ident "$c"
    if [[ -n $found_id && $found_label != "$label_value" ]]; then
      refuse "container $c exists but this script did not make it"
    fi
  done
}

guard_template_names() {
  local c
  is_listed "env $proxy_c" SYNTHETIC_STATE || refuse "SYNTHETIC_STATE is not on the template for $proxy_c"
  for c in "$dash_c" "$gate_c"; do
    is_listed "env $c" HERMES_HOME || refuse "HERMES_HOME is not on the template for $c"
  done
}

# The settings file goes to docker as it is, so every name in it must be an owner app name of the template,
# and its issuer must be the proxy's own (so that it never points at production). Values are never printed.
token_names=()
guard_settings() {
  local line name value proxy_issuer seen=" " re='^([A-Za-z_][A-Za-z0-9_]*)=(.*)$'
  [[ -f $settings ]] || refuse "stack-app.settings is missing"
  proxy_issuer=$(sed -n 's/^ISSUER = "\(.*\)"$/\1/p' "$proxy_source")
  [[ -n $proxy_issuer ]] || refuse "the issuer of the proxy cannot be read"
  while IFS= read -r line || [[ -n $line ]]; do
    if [[ -z ${line//[[:space:]]/} || $line == "#"* ]]; then
      continue
    fi
    [[ $line =~ $re ]] || refuse "stack-app.settings has a line that is not NAME=value"
    name=${BASH_REMATCH[1]} value=${BASH_REMATCH[2]}
    is_listed "env $app_c" "$name" || refuse "stack-app.settings has the name $name, which is not on the template for the owner app"
    [[ $seen != *" $name "* ]] || refuse "stack-app.settings names $name twice"
    seen+="$name "
    case $name in
      RAPHAEL_ACCESS_ISSUER)
        [[ $value == "$proxy_issuer" ]] || refuse "the issuer in stack-app.settings is not the proxy's own"
        ;;
      HERMES_*_TOKEN_FILE)
        [[ $value == "$token_dir"/* ]] || refuse "$name in stack-app.settings is not under the token folder"
        token_names+=("$name")
        ;;
    esac
  done < "$settings"
  [[ $seen == *" RAPHAEL_ACCESS_ISSUER "* ]] || refuse "stack-app.settings has no issuer"
}

work=""
make_work() {
  work=$(mktemp -d "${TMPDIR:-/tmp}/stack_up.XXXXXX")
  trap 'rm -rf "$work"' EXIT
}

# docker_build ARGS...: runs docker build quietly; the last lines of its output show on a failure.
docker_build() {
  local log=$work/build.log
  docker build "$@" > "$log" 2>&1 || {
    tail -n 5 "$log" >&2
    die "docker build failed"
  }
}

proxy_hash() { cat "$proxy_source" "$images/proxy.Dockerfile" | sha256sum | cut -c1-12; }

build_proxy() {
  local tag
  tag=$(proxy_hash)
  mkdir "$work/proxy"
  cp "$proxy_source" "$work/proxy/"
  docker_build --tag "raphael-journeys-proxy:$tag" --file "$images/proxy.Dockerfile" "$work/proxy"
  say "built proxy $tag"
}

# fetch_ca: copies ca.pem, the public file only, out of the running proxy into a folder that holds nothing else.
ca_hash=""
fetch_ca() {
  ident "$proxy_c"
  [[ -n $found_id && $found_label == "$label_value" ]] || die "the proxy container is missing"
  mkdir -p "$work/ca"
  docker cp "$found_id:$state_dir/ca.pem" "$work/ca/ca.pem" > /dev/null || die "cannot copy the CA out of the proxy"
  ca_hash=$(sha256sum "$work/ca/ca.pem" | cut -c1-12)
}

build_app() {
  plain "$1" || die "the revision is not plain"
  exists image "$fmt_id" "raphael-journeys-app-base:$1" || die "the image raphael-journeys-app-base:$1 is missing"
  docker_build --tag "raphael-journeys-app:$1-$ca_hash" --file "$images/app-ca.Dockerfile" \
    --build-arg "APP_IMAGE_DIGEST=raphael-journeys-app-base:$1" "$work/ca"
  say "built app $1-$ca_hash"
}

build_browser() {
  plain "$1" || die "the revision is not plain"
  docker_build --tag "raphael-journeys-browser:$1-$ca_hash" --file "$images/browser.Dockerfile" "$work/ca"
  say "built browser $1-$ca_hash"
}

# The platform image is built from an archive on standard input, detached: its own session, its own lock
# (held by the build process itself) and its own log, so that a build that outlives the caller's box carries on.
build_platform() {
  local commit=$1 archive=$2 dir=${TMPDIR:-/tmp} lock log
  plain "$commit" || die "the commit is not plain"
  [[ -f $archive ]] || die "the archive is not a file"
  if exists image "$fmt_id" "raphael-journeys-platform:$commit"; then
    say "image raphael-journeys-platform:$commit already exists"
    return 0
  fi
  lock=$dir/stack_up-platform-build.lock log=$dir/stack_up-platform-build.log
  exec 9> "$lock"
  if ! flock --nonblock 9; then
    exec 9>&-
    say "a platform build is already running"
    return 0
  fi
  setsid docker build --tag "raphael-journeys-platform:$commit" --build-arg "HERMES_GIT_SHA=$commit" - \
    < "$archive" > "$log" 2>&1 &
  exec 9>&-
  say "building platform $commit; log $log"
}

# --- Health: the proxy answers on its own TLS port with its own CA; the dashboard and the gateway on loopback.
# The dashboard's list of public paths is hermes_cli/dashboard_auth/public_paths.py lines 33-38 (it names
# /api/health), the route is hermes_cli/web_routers/status.py lines 112-116: /api/health answers without a
# sign-in. The gateway API server listens on 127.0.0.1:8642 (hermes_cli/owner_write_watch.py DEFAULT_PORT).
proxy_hc='python3 -c "import socket,ssl,sys;c=ssl.create_default_context(cafile='"'"'/state/ca.pem'"'"');s=c.wrap_socket(socket.create_connection(('"'"'127.0.0.1'"'"',443),5),server_hostname='"'"'synthetic-team.cloudflareaccess.com'"'"');s.sendall(b'"'"'GET /cdn-cgi/access/certs HTTP/1.1\r\nHost: synthetic-team.cloudflareaccess.com\r\nConnection: close\r\n\r\n'"'"');d=b'"'"''"'"'.join(iter(lambda:s.recv(65536),b'"'"''"'"'));sys.exit(0 if b'"'"'raphael-server2-key'"'"' in d else 1)"'
dash_hc='curl --silent --fail --max-time 5 127.0.0.1:9119/api/health > /dev/null'
gate_hc='curl --silent --fail --max-time 5 127.0.0.1:8642/health > /dev/null'
browser_hc='cd /opt/raphael-journeys && node -e "require('"'"'playwright'"'"')" && test -f /usr/local/share/ca-certificates/raphael-journeys-proxy-ca.crt'
hc_timing=(--health-interval 5s --health-timeout 10s --health-retries 3 --health-start-period 20s)

waited=0
answers() {
  local state
  state=$(docker inspect --type container --format "$fmt_state" "$1" 2> /dev/null) || return 1
  [[ $state == "true healthy" ]]
}

# wait_for ID...: until every one answers, inside the one 180-second budget, polled every 2 seconds.
wait_for() {
  local pending=("$@") still c
  while :; do
    still=()
    for c in "${pending[@]}"; do
      if ! answers "$c"; then still+=("$c"); fi
    done
    pending=("${still[@]+"${still[@]}"}")
    if ((${#pending[@]} == 0)); then return 0; fi
    if ((waited >= 180)); then return 1; fi
    sleep 2
    waited=$((waited + 2))
  done
}

# plan NAME REF: need = create (no container), replace (another image) or keep; old_id is the container's id.
need="" old_id=""
plan() {
  local have
  ident "$1"
  old_id=$found_id
  if [[ -z $old_id ]]; then
    need=create
    return 0
  fi
  have=$(docker inspect --type container --format "$fmt_config_image" "$old_id" 2> /dev/null) || have=""
  if [[ $have == "$2" ]]; then need=keep; else need=replace; fi
}

# remove NAME: removes the container of that name by its id, after its label has been read.
remove() {
  ident "$1"
  [[ -n $found_id ]] || return 0
  [[ $found_label == "$label_value" ]] || die "container $1 is not one this script made; it was left alone"
  docker rm --force "$found_id" > /dev/null || die "cannot remove $1"
  say "removed $1"
}

# start_container NAME DOCKER_RUN_ARGS...: one labelled, restarting, detached container.
start_container() {
  local name=$1
  shift
  docker run --detach --label "$label" --restart unless-stopped --name "$name" "$@" > /dev/null ||
    die "cannot create $name"
  say "created $name"
}

# create_container NAME DOCKER_RUN_ARGS...: the same container, made but not started (docker create takes the
# same arguments as docker run and is the one call that makes a container without starting it).
create_container() {
  local name=$1
  shift
  docker create --label "$label" --restart unless-stopped --name "$name" "$@" > /dev/null ||
    die "cannot create $name"
  say "created $name without starting it"
}

# running ID: the container runs now (its health is not looked at).
running() {
  local state
  state=$(docker inspect --type container --format "$fmt_state" "$1" 2> /dev/null) || return 1
  [[ $state == true* ]]
}

# tokens_ready IMAGE: sets tokens to yes when every token file that stack-app.settings names exists and is not
# empty, otherwise to no. One throw-away helper container with no network sees the token volume read-only and
# runs test -s on each file: it reads no content, writes nothing and prints nothing. The file names reach it as
# variables through --env-file, so no value is put on a command line.
tokens=""
tokens_ready() {
  local test_all="" n rc=0
  for n in "${token_names[@]+"${token_names[@]}"}"; do test_all+="test -s \"\$$n\" && "; done
  docker run --rm --label "$label" --network none --user 0 --entrypoint sh \
    --mount "type=volume,source=$tokens_v,target=$token_dir,readonly" --env-file "$settings" \
    "$1" -c "${test_all}true" > /dev/null 2>&1 || rc=$?
  case $rc in
    0) tokens=yes ;;
    1) tokens=no ;;
    *) die "cannot check the token files" ;;
  esac
}

do_versions() {
  local roles=(proxy app dashboard gateway browser) i ref
  guard_host
  guard_ownership
  for i in "${!roles[@]}"; do
    ident "${containers[i]}"
    if [[ -z $found_id ]]; then
      printf '%s none\n' "${roles[i]}"
      continue
    fi
    ref=$(docker inspect --type container --format "$fmt_config_image" "$found_id") || die "cannot read ${roles[i]}"
    printf '%s %s\n' "${roles[i]}" "${ref##*:}"
  done
}

do_converge() {
  local proxy_tag=$1 app_rev=$2 platform_tag=$3 browser_rev=$4 tag v
  local proxy_ref=raphael-journeys-proxy:$1 platform_ref=raphael-journeys-platform:$3
  local app_ref browser_ref proxy_need dash_need gate_need app_need browser_need gate_old app_old
  local proxy_id app_id gate_id maker
  local ids=()
  for tag in "$proxy_tag" "$app_rev" "$platform_tag" "$browser_rev"; do
    plain "$tag" || refuse "a tag is not plain (letters, digits, dot, dash, underscore; at most 63 characters)"
  done
  guard_host
  guard_docker
  guard_ownership
  guard_template_names
  guard_settings
  plan "$proxy_c" "$proxy_ref"
  proxy_need=$need
  plan "$dash_c" "$platform_ref"
  dash_need=$need
  plan "$gate_c" "$platform_ref"
  gate_need=$need gate_old=$old_id
  if [[ $proxy_need != keep ]]; then
    exists image "$fmt_id" "$proxy_ref" || refuse "the image $proxy_ref is missing; build it first"
  fi
  if [[ $dash_need != keep || $gate_need != keep ]]; then
    exists image "$fmt_id" "$platform_ref" || refuse "the image $platform_ref is missing; build it first"
  fi
  make_work

  # 1 and 2: the network and the volumes, only when missing. A volume is never removed or emptied here.
  if ! exists network "$fmt_name" "$network"; then
    docker network create --driver bridge "$network" > /dev/null || die "cannot create the network"
    say "created the network"
  fi
  for v in "${volumes[@]}"; do
    if ! exists volume "$fmt_name" "$v"; then
      docker volume create "$v" > /dev/null || die "cannot create the volume $v"
      say "created the volume $v"
    fi
  done

  # 3: the proxy, then wait until it is healthy (its first start writes the CA).
  printf 'SYNTHETIC_STATE=%s\n' "$state_dir" > "$work/proxy.env"
  printf 'HERMES_HOME=%s\n' "$data_dir" > "$work/platform.env"
  if [[ $proxy_need == replace ]]; then remove "$proxy_c"; fi
  if [[ $proxy_need != keep ]]; then
    start_container "$proxy_c" --network "$network" \
      --network-alias workspace.synthetic.test --network-alias synthetic-team.cloudflareaccess.com \
      --mount "type=volume,source=$proxy_v,target=$state_dir" --env-file "$work/proxy.env" \
      --health-cmd "$proxy_hc" "${hc_timing[@]}" "$proxy_ref"
  fi
  ident "$proxy_c"
  proxy_id=$found_id
  wait_for "$proxy_id" || die "the proxy does not answer"

  # 4 and 5: the CA leaves the proxy as ca.pem only; the owner app and browser layers carry its hash.
  fetch_ca
  app_ref=raphael-journeys-app:$app_rev-$ca_hash
  browser_ref=raphael-journeys-browser:$browser_rev-$ca_hash
  if ! exists image "$fmt_id" "$app_ref"; then build_app "$app_rev"; fi
  if ! exists image "$fmt_id" "$browser_ref"; then build_browser "$browser_rev"; fi

  # 6: the dashboard.
  if [[ $dash_need == replace ]]; then remove "$dash_c"; fi
  if [[ $dash_need != keep ]]; then
    start_container "$dash_c" --network "$network" --network-alias dashboard \
      --mount "type=volume,source=$home_v,target=$data_dir" \
      --mount "type=volume,source=$tokens_v,target=$token_dir" --env-file "$work/platform.env" \
      --health-cmd "$dash_hc" "${hc_timing[@]}" "$platform_ref" dashboard --host 0.0.0.0 --port 9119 --no-open
  fi

  # 7: the owner app, then the gateway. The gateway joins the owner app's network namespace, so it is made after
  # the app and made again whenever the app is made again or does not name the current app.
  plan "$app_c" "$app_ref"
  app_need=$need app_old=$old_id
  if [[ $gate_need == keep ]]; then
    if [[ $app_need != keep ]]; then
      gate_need=replace
    elif [[ $(docker inspect --type container --format "$fmt_network_mode" "$gate_old" 2> /dev/null) != "container:$app_old" ]]; then
      gate_need=replace
    elif running "$gate_old" && ! running "$app_old"; then
      gate_need=replace
    fi
  fi
  # One read-only look at the token files, only when the owner app or the gateway is about to be made or started.
  tokens=yes
  if [[ $app_need != keep || $gate_need != keep ]] || ! running "$app_old" || ! running "$gate_old"; then
    tokens_ready "$platform_ref"
  fi
  # While a token file is missing the two are only made: the first reset issues the tokens and starts them.
  maker=start_container
  if [[ $tokens != yes ]]; then maker=create_container; fi
  if [[ $gate_need == replace ]]; then remove "$gate_c"; fi
  if [[ $app_need == replace ]]; then remove "$app_c"; fi
  if [[ $app_need != keep ]]; then
    "$maker" "$app_c" --network "$network" --network-alias workspace \
      --mount "type=volume,source=$tokens_v,target=$token_dir" --env-file "$settings" "$app_ref"
  fi
  ident "$app_c"
  app_id=$found_id
  if [[ $tokens == yes && $app_need == keep ]] && ! running "$app_id"; then
    docker start "$app_id" > /dev/null || die "cannot start $app_c"
    say "started $app_c"
  fi
  if [[ $gate_need != keep ]]; then
    "$maker" "$gate_c" --network "container:$app_id" \
      --mount "type=volume,source=$home_v,target=$data_dir" --env-file "$work/platform.env" \
      --health-cmd "$gate_hc" "${hc_timing[@]}" "$platform_ref" gateway run
  fi
  ident "$gate_c"
  gate_id=$found_id
  if [[ $tokens == yes && $gate_need == keep ]] && ! running "$gate_id"; then
    docker start "$gate_id" > /dev/null || die "cannot start $gate_c"
    say "started $gate_c"
  fi

  # 8: the browser.
  plan "$browser_c" "$browser_ref"
  browser_need=$need
  if [[ $browser_need == replace ]]; then remove "$browser_c"; fi
  if [[ $browser_need != keep ]]; then
    start_container "$browser_c" --network "$network" --health-cmd "$browser_hc" "${hc_timing[@]}" "$browser_ref"
  fi

  # 9: all five answer inside the one budget; while a token file is missing, the proxy, the dashboard and the
  # browser only.
  if [[ $tokens == yes ]]; then
    for v in "$proxy_c" "$app_c" "$dash_c" "$gate_c" "$browser_c"; do
      ident "$v"
      ids+=("$found_id")
    done
  else
    for v in "$proxy_c" "$dash_c" "$browser_c"; do
      ident "$v"
      ids+=("$found_id")
    done
  fi
  wait_for "${ids[@]}" || die "no answer after 180 seconds from at least one container; run converge again"
  if [[ $tokens == yes ]]; then
    say "the stack answers"
  else
    say "the first reset starts the owner app and the gateway"
  fi
}

(($# >= 1)) || usage
case $1 in
  versions)
    (($# == 1)) || usage
    do_versions
    ;;
  build)
    (($# >= 2)) || usage
    case $2 in
      proxy) (($# == 2)) || usage ;;
      app | browser) (($# == 3)) || usage ;;
      platform) (($# == 4)) || usage ;;
      *) usage ;;
    esac
    guard_host
    guard_docker
    case $2 in
      proxy)
        make_work
        build_proxy
        ;;
      app | browser)
        make_work
        fetch_ca
        "build_$2" "$3"
        ;;
      platform) build_platform "$3" "$4" ;;
    esac
    ;;
  converge)
    (($# == 5)) || usage
    shift
    do_converge "$@"
    ;;
  *) usage ;;
esac
