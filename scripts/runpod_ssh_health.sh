#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  scripts/runpod_ssh_health.sh user@ssh.runpod.io [key_path]
  scripts/runpod_ssh_health.sh ssh user@ssh.runpod.io -i ~/.ssh/id_ed25519 [-p PORT]

Checks the actual RunPod SSH connection with a clean SSH config. It does not
query, create, stop, or modify pods.
EOF
}

target=""
key_path="${RUNPOD_SSH_KEY:-$HOME/.ssh/id_ed25519}"
port="${RUNPOD_SSH_PORT:-}"

if [[ $# -eq 0 ]]; then
  usage
  exit 2
fi

if [[ "$1" == "ssh" ]]; then
  shift
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -i | -p)
        if [[ $# -lt 2 ]]; then
          printf 'missing value for %s\n' "$1" >&2
          exit 2
        fi
        if [[ "$1" == "-i" ]]; then
          key_path="$2"
        else
          port="$2"
        fi
        shift 2
        ;;
      *@*)
        target="$1"
        shift
        ;;
      *)
        shift
        ;;
    esac
  done
else
  target="$1"
  if [[ $# -ge 2 ]]; then
    key_path="$2"
  fi
fi

if [[ -z "$target" || "$target" != *@* ]]; then
  printf 'missing SSH target; expected user@host from the RunPod Connect tab\n' >&2
  usage
  exit 2
fi

if [[ ! -r "$key_path" ]]; then
  printf 'SSH key is not readable: %s\n' "$key_path" >&2
  exit 2
fi

remote_command=${RUNPOD_SSH_HEALTH_COMMAND:-'printf "__RUNPOD_SSH_HEALTH_OK__\n"; hostname; if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi -L; fi'}
tmp_output="$(mktemp "${TMPDIR:-/tmp}/runpod-ssh-health.XXXXXX")"
trap 'rm -f "$tmp_output"' EXIT

ssh_args=(
  -F /dev/null
  -tt
  -o BatchMode=yes
  -o ConnectTimeout=12
  -o ConnectionAttempts=1
  -o ServerAliveInterval=4
  -o ServerAliveCountMax=1
  -o IdentitiesOnly=yes
  -o ControlMaster=no
  -o ControlPath=none
  -o PreferredAuthentications=publickey
  -o PasswordAuthentication=no
  -o KbdInteractiveAuthentication=no
  -o NumberOfPasswordPrompts=0
  -o StrictHostKeyChecking=accept-new
  -i "$key_path"
)

if [[ -n "$port" ]]; then
  ssh_args+=(-p "$port")
fi

set +e
ssh "${ssh_args[@]}" "$target" "$remote_command" | tr -d '\r' | tee "$tmp_output"
ssh_status=${PIPESTATUS[0]}
set -e

if [[ "$ssh_status" -ne 0 ]]; then
  printf 'RUNPOD_SSH_HEALTH_FAILED ssh_exit=%s target=%s\n' "$ssh_status" "$target" >&2
  exit "$ssh_status"
fi

if ! grep -qx '__RUNPOD_SSH_HEALTH_OK__' "$tmp_output"; then
  printf 'RUNPOD_SSH_HEALTH_FAILED sentinel_missing target=%s\n' "$target" >&2
  exit 1
fi

printf 'RUNPOD_SSH_HEALTH_READY target=%s\n' "$target"
