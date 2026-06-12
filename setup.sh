#!/usr/bin/env bash

SCRIPT_PATH="${BASH_SOURCE[0]:-$0}"
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$SCRIPT_PATH")" && pwd)
ROOT=$SCRIPT_DIR

case "$(uname -s 2>/dev/null || echo unknown)" in
  MINGW*|MSYS*|CYGWIN*) TARGET="$ROOT/.build/xtask/debug/xtask.exe" ;;
  *) TARGET="$ROOT/.build/xtask/debug/xtask" ;;
esac

STAMP="$ROOT/.build/xtask/sipp.stamp"
BIN_DIR="$ROOT/.build/bin"
ENV_SCRIPT="$BIN_DIR/sipp-env.sh"

is_sourced=0
if (return 0 2>/dev/null); then
  is_sourced=1
fi

needs_build=0
if [ ! -x "$TARGET" ] || [ ! -f "$STAMP" ]; then
  needs_build=1
elif find "$ROOT/xtask/src" "$ROOT/xtask/Cargo.toml" "$ROOT/Cargo.toml" "$ROOT/Cargo.lock" "$ROOT/.cargo/config.toml" -newer "$STAMP" -print -quit 2>/dev/null | grep -q .; then
  needs_build=1
fi

if [ "$needs_build" = "1" ]; then
  (cd "$ROOT" && cargo build --target-dir .build/xtask --package xtask --quiet)
  build_status=$?
  if [ "$build_status" -ne 0 ]; then
    if [ "$is_sourced" = "1" ]; then
      return "$build_status"
    fi
    exit "$build_status"
  fi
  mkdir -p "$(dirname "$STAMP")"
  : > "$STAMP"
fi

case ":${PATH:-}:" in
  *:"$BIN_DIR":*) ;;
  *) export PATH="$BIN_DIR${PATH:+:$PATH}" ;;
esac

"$TARGET" setup "$@"
setup_status=$?
if [ "$setup_status" -ne 0 ]; then
  if [ "$is_sourced" = "1" ]; then
    return "$setup_status"
  fi
  exit "$setup_status"
fi

if [ -f "$ENV_SCRIPT" ]; then
  if [ "$is_sourced" = "1" ]; then
    # shellcheck disable=SC1090
    . "$ENV_SCRIPT"
    hash -r 2>/dev/null || true
    printf '\nsipp is active in this shell session.\n'
  else
    printf '\nTo use sipp in this shell, run:\n  source "%s"\n' "$ENV_SCRIPT"
  fi
fi
