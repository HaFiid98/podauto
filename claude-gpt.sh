#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Claude Code + AgentRouter + gpt-5.6-sol
#
# Usage:
#   chmod +x agentrouter-gpt.sh
#   ./agentrouter-gpt.sh
#
# IMPORTANT:
# Create a NEW AgentRouter API key.
# The key previously pasted into chat should be revoked.
# ============================================================

# ------------------------------------------------------------
# 1. YOUR NEW AGENTROUTER KEY
# ------------------------------------------------------------
KEY="sk-zhZ1DrV05kqtpb8BP7pWiMu7viNGMUVs73kTYZSemUYY1WVi"

# ------------------------------------------------------------
# 2. AGENTROUTER ANTHROPIC-COMPATIBLE ENDPOINT
# ------------------------------------------------------------
BASE_URL="https://co.agentrouter.org"

# ------------------------------------------------------------
# 3. MODEL FROM YOUR SCREENSHOT
# ------------------------------------------------------------
MODEL="gpt-5.6-sol"


# ============================================================
# Checks
# ============================================================

if [[ "$KEY" == "PUT_YOUR_NEW_AGENTROUTER_KEY_HERE" || -z "$KEY" ]]; then
    echo
    echo "ERROR: Put your NEW AgentRouter API key in:"
    echo
    echo 'KEY="..."'
    echo
    exit 1
fi

if ! command -v claude >/dev/null 2>&1; then
    echo "Claude Code not found."
    echo "Installing Claude Code..."
    npm install -g @anthropic-ai/claude-code@latest
fi


# ============================================================
# Write Claude Code configuration
# ============================================================

mkdir -p "$HOME/.claude"

node - "$KEY" "$BASE_URL" "$MODEL" <<'NODE'
const [key, baseUrl, model] = process.argv.slice(2);

const fs = require("fs");

const path = process.env.HOME + "/.claude/settings.json";

let cfg = {};

if (fs.existsSync(path)) {
    try {
        cfg = JSON.parse(fs.readFileSync(path, "utf8"));
    } catch {
        console.log("Existing settings.json is invalid.");
        console.log("Creating a new configuration.");
        cfg = {};
    }
}

cfg.env = cfg.env || {};

// AgentRouter
cfg.env.ANTHROPIC_BASE_URL = baseUrl;
cfg.env.ANTHROPIC_AUTH_TOKEN = key;

// Do not let Claude Code use a different API key.
cfg.env.ANTHROPIC_API_KEY = "";

// Requested model
cfg.env.ANTHROPIC_MODEL = model;

// Force the Claude Code model aliases to the requested model.
cfg.env.ANTHROPIC_DEFAULT_OPUS_MODEL = model;
cfg.env.ANTHROPIC_DEFAULT_SONNET_MODEL = model;
cfg.env.ANTHROPIC_DEFAULT_HAIKU_MODEL = model;

// Subagents
cfg.env.CLAUDE_CODE_SUBAGENT_MODEL = model;

// Let AgentRouter advertise models if it supports them.
cfg.env.CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY = "1";

fs.writeFileSync(
    path,
    JSON.stringify(cfg, null, 2) + "\n"
);

console.log("");
console.log("Claude Code configuration written:");
console.log("  Base URL :", baseUrl);
console.log("  Model    :", model);
console.log("");
NODE


# ============================================================
# Export environment variables for THIS session
# ============================================================

export ANTHROPIC_BASE_URL="$BASE_URL"
export ANTHROPIC_AUTH_TOKEN="$KEY"

# Must remain empty.
export ANTHROPIC_API_KEY=""

export ANTHROPIC_MODEL="$MODEL"

export ANTHROPIC_DEFAULT_OPUS_MODEL="$MODEL"
export ANTHROPIC_DEFAULT_SONNET_MODEL="$MODEL"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="$MODEL"

export CLAUDE_CODE_SUBAGENT_MODEL="$MODEL"

export CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY="1"


# ============================================================
# Display configuration
# ============================================================

echo
echo "============================================================"
echo " Claude Code + AgentRouter"
echo "============================================================"
echo
echo " Base URL : $ANTHROPIC_BASE_URL"
echo " Model    : $ANTHROPIC_MODEL"
echo
echo " Starting Claude Code..."
echo


# ============================================================
# Start Claude Code
# ============================================================

exec claude