#!/bin/bash
# Generate release notes from conventional commits
# Usage: ./generate-release-notes.sh <from-tag> <to-tag>
# If no tags, uses all commits

set -euo pipefail

FROM="${1:-}"
TO="${2:-HEAD}"

if [ -z "$FROM" ] || [ "$FROM" = "none" ]; then
  # No previous tag — use all commits
  RANGE="$TO"
  echo "## Initial Release"
  echo ""
  echo "Full commit history:"
  echo ""
else
  RANGE="${FROM}..${TO}"
fi

FEATURES=$(git log "$RANGE" --no-merges --pretty=format:"- %s (%an)" --grep="^feat" 2>/dev/null || true)
FIXES=$(git log "$RANGE" --no-merges --pretty=format:"- %s (%an)" --grep="^fix" 2>/dev/null || true)
CHORES=$(git log "$RANGE" --no-merges --pretty=format:"- %s (%an)" --grep="^chore\|^refactor\|^ci" 2>/dev/null || true)
TESTS=$(git log "$RANGE" --no-merges --pretty=format:"- %s (%an)" --grep="^test" 2>/dev/null || true)
DOCS=$(git log "$RANGE" --no-merges --pretty=format:"- %s (%an)" --grep="^docs" 2>/dev/null || true)

OUTPUT=""

if [ -n "$FEATURES" ]; then
  OUTPUT+="### 🚀 Features\n\n$FEATURES\n\n"
fi

if [ -n "$FIXES" ]; then
  OUTPUT+="### 🐛 Bug Fixes\n\n$FIXES\n\n"
fi

if [ -n "$TESTS" ]; then
  OUTPUT+="### ✅ Tests\n\n$TESTS\n\n"
fi

if [ -n "$DOCS" ]; then
  OUTPUT+="### 📚 Documentation\n\n$DOCS\n\n"
fi

if [ -n "$CHORES" ]; then
  OUTPUT+="### 🔧 Maintenance\n\n$CHORES\n\n"
fi

if [ -z "$OUTPUT" ]; then
  OUTPUT="### Changes\n\nSee individual commits for details.\n"
fi

echo -e "$OUTPUT"
