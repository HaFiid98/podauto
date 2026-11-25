#!/usr/bin/env bash
set -Eeuo pipefail

KEY="sk-ZSAo1zu0veoPSE1XlJzYMjPU4BaQGx8ywReuvAG88guMT8JH"
BASE_URL="https://agentrouter.org"
MODEL="gpt-6.1-sol"

echo "[1/4] Checking API key..."
if [[ "$KEY" == "PASTE_YOUR_NEW_AGENTROUTER_KEY_HERE" || "$KEY" != sk-* ]]; then
    echo "ERROR: Set your AgentRouter API key in KEY."
    exit 1
fi

echo "[2/4] Checking Claude Code..."
if ! command -v claude >/dev/null 2>&1; then
    echo "ERROR: Claude Code is not installed or not in PATH."
    echo "Install it using the official Claude Code installation method."
    exit 1
fi

echo "[3/4] Writing configuration..."
mkdir -p "$HOME/.claude"
chmod 700 "$HOME/.claude"

node - "$KEY" "$BASE_URL" "$MODEL" <<'NODE'
const fs = require("fs");
const path = require("path");

const [key, baseUrl, model] = process.argv.slice(2);
const file = path.join(process.env.HOME, ".claude", "settings.json");
let cfg = {};

if (fs.existsSync(file)) {
    try {
        cfg = JSON.parse(fs.readFileSync(file, "utf8"));
    } catch (e) {
        cfg = {};
    }
}

cfg.env = cfg.env || {};
cfg.env.ANTHROPIC_BASE_URL = baseUrl;
cfg.env.ANTHROPIC_AUTH_TOKEN = key;
delete cfg.env.ANTHROPIC_API_KEY;

// Map all internal Claude Code model tiers to gpt-6.1-sol
cfg.env.ANTHROPIC_MODEL = model;
cfg.env.ANTHROPIC_DEFAULT_MODEL = model;
cfg.env.ANTHROPIC_DEFAULT_OPUS_MODEL = model;
cfg.env.ANTHROPIC_DEFAULT_SONNET_MODEL = model;
cfg.env.ANTHROPIC_DEFAULT_HAIKU_MODEL = model;
cfg.env.CLAUDE_CODE_SUBAGENT_MODEL = model;
cfg.env.CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY = "1";

cfg.model = model;

fs.writeFileSync(file, JSON.stringify(cfg, null, 2) + "\n", {
    mode: 0o600
});
fs.chmodSync(file, 0o600);
NODE

echo "[4/4] Starting Claude Code..."
export ANTHROPIC_BASE_URL="$BASE_URL"
export ANTHROPIC_AUTH_TOKEN="$KEY"
unset ANTHROPIC_API_KEY
export ANTHROPIC_MODEL="$MODEL"
export ANTHROPIC_DEFAULT_MODEL="$MODEL"
export ANTHROPIC_DEFAULT_OPUS_MODEL="$MODEL"
export ANTHROPIC_DEFAULT_SONNET_MODEL="$MODEL"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="$MODEL"
export CLAUDE_CODE_SUBAGENT_MODEL="$MODEL"
export CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1

exec claude --model "$MODEL"
