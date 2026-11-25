#!/usr/bin/env bash
set -Eeuo pipefail

KEY="sk-ZSAo1zu0veoPSE1XlJzYMjPU4BaQGx8ywReuvAG88guMT8JH"
BASE_URL="https://agentrouter.org/v1"
MODEL="gpt-6.1-sol"

echo "[1/4] Checking API key..."
if [[ "$KEY" == "PASTE_YOUR_NEW_AGENTROUTER_KEY_HERE" || "$KEY" != sk-* ]]; then
    echo "ERROR: Set your AgentRouter API key in KEY."
    exit 1
fi

echo "[2/4] Checking Codex CLI..."
if ! command -v codex >/dev/null 2>&1; then
    echo "ERROR: Codex CLI is not installed or not in PATH."
    echo "Install it via: npm install -g @openai/codex"
    exit 1
fi

echo "[3/4] Writing ~/.codex/config.toml..."
mkdir -p "$HOME/.codex"
chmod 700 "$HOME/.codex"

cat > "$HOME/.codex/config.toml" <<EOF
model = "$MODEL"
model_provider = "agentrouter"

[model_providers.agentrouter]
name = "AgentRouter"
base_url = "$BASE_URL"
experimental_bearer_token = "$KEY"
wire_api = "responses"
requires_openai_auth = false
EOF

chmod 600 "$HOME/.codex/config.toml"

echo "[4/4] Starting Codex CLI..."
export AGENTROUTER_API_KEY="$KEY"

exec codex --model "$MODEL" resume
