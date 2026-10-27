#!/usr/bin/env bash
# Start one run of the scripted owner journeys (card P-3) as the transient user service hermes-journeys.
# Takes the run lock without waiting, refuses while the unit is still loaded (as the release unit does,
# hermes_cli/release_unit.py), then asks the user manager for the service with a hard time limit.
# Refusals print one "Refused: ..." line and exit 1. Nothing retries and nothing is printed but those lines.
# Only names and addresses of the stack travel to the unit; no secret value is ever read or passed on.
set -euo pipefail
umask 077

UNIT=hermes-journeys.service
RUN_MAX_SECONDS=2700
# Names the run reads from its environment (see run_once.sh and the two scripts it calls).
PASS_ON=(PATH HERMES_HOME JOURNEY_STACK_HOST JOURNEY_PRODUCTION_ORIGIN JOURNEY_MODEL_TOKEN_FILE
  JOURNEY_BASE_URL JOURNEY_PAGE_TEXT JOURNEY_RUNNER_SSH)

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUN_DIR=${HERMES_HOME:-$HOME/.hermes}/journeys
LOCK=$RUN_DIR/run.lock

refuse() {
  printf 'Refused: %s\n' "$1"
  exit 1
}

command -v systemd-run > /dev/null 2>&1 || refuse "systemd-run is missing."
command -v systemctl > /dev/null 2>&1 || refuse "systemctl is missing."

mkdir -p "$RUN_DIR" 2> /dev/null && exec 9>> "$LOCK" ||
  refuse "the folder of the run lock cannot be used."
flock -n 9 || refuse "a journey run is still at work (the run lock is held)."

# A unit of that name that is loaded and not at rest is a run at work, or one starting or stopping.
listed=$(systemctl --user list-units --all --full --plain --no-legend --no-pager "$UNIT" 2> /dev/null) ||
  refuse "the user manager could not be asked about hermes-journeys."
while read -r name _load active _rest; do
  if [[ $name == "$UNIT" && $active != inactive && $active != failed ]]; then
    refuse "the unit hermes-journeys is still loaded."
  fi
done <<< "$listed"

setenv=()
for name in "${PASS_ON[@]}"; do
  if [[ -n ${!name-} ]]; then setenv+=("--setenv=$name=${!name}"); fi
done

# The run takes the lock itself and holds it to its end, so this call lets go of it first; a second
# start in between is refused by the user manager, which loads one unit of that name only.
exec 9>&-
if ! failure=$(systemd-run --user "--unit=$UNIT" --collect "--property=RuntimeMaxSec=$RUN_MAX_SECONDS" \
  "${setenv[@]}" -- "$HERE/run_once.sh" 2>&1 > /dev/null); then
  printf 'Refused: the unit hermes-journeys could not be started.\n%s\n' "$failure"
  exit 1
fi
printf 'Started the unit hermes-journeys (time limit %s seconds).\n' "$RUN_MAX_SECONDS"
