#!/usr/bin/env bash
# Push all three scaffold projects as new public GitHub repos.
# Requires: gh CLI authenticated (gh auth login)
set -euo pipefail

USER="jyotisaini-3"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

push_scaffold() {
  local name="$1"
  local dir="$SCRIPT_DIR/$name"
  echo "\n=== $name ==="
  cd "$dir"
  git init -b main
  git add .
  git commit -m "feat: initial scaffold — $name"
  # Create repo and push (gh CLI)
  gh repo create "$USER/$name" --public --source=. --remote=origin --push
  echo "✓ https://github.com/$USER/$name"
}

push_scaffold hopper-warp-gemm
push_scaffold gpu-roofline-analysis
push_scaffold powered-descent-gpu

echo "\nAll done! Now pin these + your forks on https://github.com/$USER"
