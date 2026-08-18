#!/usr/bin/env bash
set -euo pipefail

# ============================================================
#  AgentRouter + Claude Code - zero-setup launcher
#  STEP 1: paste key below
#  STEP 2: run  bash agentrouter-launch.sh
# ============================================================

KEY="sk-zhZ1DrV05kqtpb8BP7pWiMu7viNGMUVs73kTYZSemUYY1WVi"          # https://agentrouter.org/console/token
BASE_URL="${ANTHROPIC_BASE_URL:-https://agentrouter.org/}"

# ---- nothing below needs editing ----
[[ "$KEY" =~ ^sk- && "$KEY" != "sk-paste-your-key-here" ]] || { echo "Put your key in the KEY line."; exit 1; }
command -v claude >/dev/null 2>&1 || npm install -g @anthropic-ai/claude-code

mkdir -p "$HOME/.claude"
node - "$KEY" "$BASE_URL" <<'NODE'
const [key, baseUrl] = process.argv.slice(2);
const fs = require('fs');
const p = process.env.HOME + '/.claude/settings.json';
const cfg = fs.existsSync(p) ? JSON.parse(fs.readFileSync(p, 'utf8')) : {};
cfg.env = cfg.env || {};
cfg.env.ANTHROPIC_BASE_URL = baseUrl;
cfg.env.ANTHROPIC_AUTH_TOKEN = key;
cfg.env.ANTHROPIC_API_KEY = key;
cfg.env.CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY = '1';
fs.writeFileSync(p, JSON.stringify(cfg, null, 2) + '\n');
NODE

export ANTHROPIC_BASE_URL="$BASE_URL" ANTHROPIC_AUTH_TOKEN="$KEY" ANTHROPIC_API_KEY="$KEY"
exec claude