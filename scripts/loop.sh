#!/usr/bin/env bash
# 자동 루프 1회전: 체크포인트 판정 → docs/status.md → 사이트 게시.
#   scripts/loop.sh            # fast만
#   scripts/loop.sh --live     # LLM/Tavily 포함(비용 표시)
set -e
cd "$(dirname "$0")/.."
.venv/bin/python scripts/checkpoints.py "$@"
scripts/publish_site.sh
