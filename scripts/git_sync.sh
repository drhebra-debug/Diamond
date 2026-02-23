#!/usr/bin/env bash
# Simple Git sync script for Linux/macOS
set -euo pipefail

REPO_PATH=${1:-$(pwd)}
BRANCH=${2:-main}
REMOTE=${3:-origin}
DRYRUN=${DRYRUN:-false}

LOG_FILE="$REPO_PATH/git_sync.log"

timestamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
echo "$(timestamp) - Starting sync. RepoPath=$REPO_PATH Branch=$BRANCH Remote=$REMOTE DRYRUN=$DRYRUN" >> "$LOG_FILE"

cd "$REPO_PATH"

echo "$(timestamp) - fetching $REMOTE" >> "$LOG_FILE"
if ! git fetch "$REMOTE" >> "$LOG_FILE" 2>&1; then
  echo "$(timestamp) - git fetch failed" >> "$LOG_FILE"
fi

echo "$(timestamp) - pulling $REMOTE/$BRANCH (rebase)" >> "$LOG_FILE"
if ! git pull --rebase "$REMOTE" "$BRANCH" >> "$LOG_FILE" 2>&1; then
  echo "$(timestamp) - git pull --rebase failed, aborting rebase and trying merge" >> "$LOG_FILE"
  git rebase --abort || true
  if ! git pull "$REMOTE" "$BRANCH" >> "$LOG_FILE" 2>&1; then
    echo "$(timestamp) - fallback git pull failed" >> "$LOG_FILE"
  else
    echo "$(timestamp) - fallback git pull succeeded" >> "$LOG_FILE"
  fi
else
  echo "$(timestamp) - git pull --rebase succeeded" >> "$LOG_FILE"
fi

STATUS="$(git status --porcelain)"
if [ -n "$STATUS" ]; then
  echo "$(timestamp) - Local changes detected" >> "$LOG_FILE"
  if [ "$DRYRUN" = "true" ]; then
    echo "$(timestamp) - Dry run enabled, skipping commit/push" >> "$LOG_FILE"
  else
    git add -A
    if git commit -m "Auto-sync: $(timestamp)" >> "$LOG_FILE" 2>&1; then
      echo "$(timestamp) - Committed local changes" >> "$LOG_FILE"
    else
      echo "$(timestamp) - Nothing to commit or commit failed" >> "$LOG_FILE"
    fi

    if git push "$REMOTE" "HEAD:$BRANCH" >> "$LOG_FILE" 2>&1; then
      echo "$(timestamp) - git push succeeded" >> "$LOG_FILE"
    else
      echo "$(timestamp) - git push failed" >> "$LOG_FILE"
    fi
  fi
else
  echo "$(timestamp) - No local changes to commit." >> "$LOG_FILE"
fi

exit 0
