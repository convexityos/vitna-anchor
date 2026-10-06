#!/bin/sh
# vitna-anchor's installer for Linux on x86-64 (gate A12). One command:
#
#   curl -fsSL https://github.com/convexityos/vitna-anchor/releases/latest/download/install.sh | sh
#
# It fetches the engine built for this machine and checks it against the
# SHA-256 the release wrote into this script; asks the engine which model of
# its catalogue fits the machine (vitna-anchor plan); fetches that model's
# files from Hugging Face at their pinned revisions, resuming any part already
# fetched; has the engine check each against its pinned SHA-256 (vitna-anchor
# verify); and starts the server, whose chat page is at http://127.0.0.1:8765/.
# Run again, it fetches only what is missing. $VITNA_HOME/serve.sh starts the
# server again later.
#
#   VITNA_HOME       where it installs (default: ~/.vitna-anchor)
#   VITNA_MODEL      a model of the catalogue in place of the one it chooses
#   VITNA_PORT       the server's port (default: 8765)
#   VITNA_NO_START   1 to stop once everything is fetched and checked
#
# For tests before a release exists: VITNA_ENGINE_URL and VITNA_ENGINE_SHA256
# fetch the engine from elsewhere, as CI's install check does.
#
# The repository's copy is a template: the release fills in its version and
# the engine's SHA-256. Everything runs from main, at the end, so a script cut
# short in its download runs nothing.

set -eu

VERSION="@VERSION@"
ENGINE_SHA256="@SHA256_LINUX_X86_64@"

say() { printf '%s\n' "$*"; }
die() {
  printf 'vitna-anchor: %s\n' "$*" >&2
  exit 1
}

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | cut -d ' ' -f 1
  else
    shasum -a 256 "$1" | cut -d ' ' -f 1
  fi
}

size_of() {
  wc -c <"$1" | tr -d ' '
}

# fetch <url> <file> <size>: the file, resuming what is there, unless it is
# already whole; what is longer than the file should be is fetched anew.
fetch() {
  mkdir -p "$(dirname "$2")"
  if [ -f "$2" ]; then
    have=$(size_of "$2")
    if [ "$have" -eq "$3" ]; then
      return 0
    fi
    if [ "$have" -gt "$3" ]; then
      rm -f "$2"
    fi
  fi
  curl -fL --retry 5 --retry-delay 3 -C - -o "$2" "$1" </dev/null || die "could not fetch $1"
}

# A word for a shell script, in single quotes.
quote() {
  printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"
}

main() {
  command -v curl >/dev/null 2>&1 || die "this installer needs curl"
  case "$(uname -s) $(uname -m)" in
    "Linux x86_64") asset=vitna-anchor-linux-x86_64 ;;
    *) die "there is no build for $(uname -s) on $(uname -m) yet; build it from source: https://github.com/convexityos/vitna-anchor" ;;
  esac
  home="${VITNA_HOME:-$HOME/.vitna-anchor}"
  port="${VITNA_PORT:-8765}"
  url="${VITNA_ENGINE_URL:-https://github.com/convexityos/vitna-anchor/releases/download/$VERSION/$asset}"
  want="${VITNA_ENGINE_SHA256:-$ENGINE_SHA256}"
  case "$want$url" in
    *@*) die "this is the installer's template, which a release fills in: run the one attached to a release" ;;
  esac
  mkdir -p "$home/bin" "$home/models"
  engine="$home/bin/vitna-anchor"

  say "Fetching the engine: $url"
  curl -fL --retry 5 --retry-delay 3 -o "$engine.part" "$url" </dev/null || die "could not fetch the engine"
  got=$(sha256_of "$engine.part")
  [ "$got" = "$want" ] || {
    rm -f "$engine.part"
    die "the engine's SHA-256 is $got, and the release says $want"
  }
  chmod +x "$engine.part"
  mv -f "$engine.part" "$engine"

  plan="$home/plan.tsv"
  if [ -n "${VITNA_MODEL:-}" ]; then
    "$engine" plan --dir "$home/models" --model "$VITNA_MODEL" >"$plan" || die "no plan for this machine"
  else
    "$engine" plan --dir "$home/models" >"$plan" || die "no plan for this machine"
  fi
  tab=$(printf '\t')
  model=$(grep "^model$tab" "$plan" | cut -f 2)
  say ""
  say "Model: $(grep "^title$tab" "$plan" | cut -f 2)"
  say "  $(grep "^note$tab" "$plan" | cut -f 2)"
  say "  $(grep "^why$tab" "$plan" | cut -f 2)"
  need=$(grep "^fetch$tab" "$plan" | cut -f 2)
  say "  $(awk -v b="$need" 'BEGIN { printf "%.1f GB to fetch", b / 1e9 }')"
  say ""

  # Each file, then the check; a file that fails it is fetched anew, once.
  for round in 1 2; do
    while IFS="$tab" read -r kind file_url path size sha; do
      [ "$kind" = file ] || continue
      say "Fetching $path"
      fetch "$file_url" "$home/models/$path" "$size"
    done <"$plan"
    say "Checking every file against its pinned SHA-256"
    if "$engine" verify --dir "$home/models" --model "$model" >"$home/verify.tsv"; then
      break
    fi
    [ "$round" = 1 ] || {
      cat "$home/verify.tsv" >&2
      die "the files are not the pinned files"
    }
    while IFS="$tab" read -r verdict path why; do
      [ "$verdict" = bad ] || continue
      say "  $path: $why; fetching it anew"
      rm -f "$home/models/$path"
    done <"$home/verify.tsv"
  done

  # The server's arguments, as the plan gives them, and a script that starts it again.
  set -f
  serve_line=$(grep "^serve$tab" "$plan")
  old_ifs=$IFS
  IFS=$tab
  # shellcheck disable=SC2086
  set -- $serve_line
  IFS=$old_ifs
  shift
  {
    printf '#!/bin/sh\n# Starts the server vitna-anchor installed, on port $VITNA_PORT or 8765.\nexec %s serve' "$(quote "$engine")"
    for arg in "$@"; do printf ' %s' "$(quote "$arg")"; done
    printf ' --port "${VITNA_PORT:-8765}"\n'
  } >"$home/serve.sh"
  chmod +x "$home/serve.sh"
  say ""
  say "Installed in $home. $home/serve.sh starts the server."
  say "OpenAI clients: http://127.0.0.1:$port/v1   Anthropic clients: http://127.0.0.1:$port   model: $model"
  if [ "${VITNA_NO_START:-}" = 1 ]; then
    return 0
  fi

  say "Starting the server; its chat page is at http://127.0.0.1:$port/ (Ctrl+C stops it)"
  "$engine" serve "$@" --port "$port" &
  pid=$!
  trap 'kill "$pid" 2>/dev/null' INT TERM
  i=0
  while [ "$i" -lt 900 ] && kill -0 "$pid" 2>/dev/null; do
    if curl -fs "http://127.0.0.1:$port/v1/health" >/dev/null 2>&1; then
      if command -v xdg-open >/dev/null 2>&1 && [ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]; then
        xdg-open "http://127.0.0.1:$port/" >/dev/null 2>&1 || true
      fi
      break
    fi
    sleep 1
    i=$((i + 1))
  done
  wait "$pid"
}

main "$@"
