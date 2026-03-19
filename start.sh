#!/usr/bin/env bash
# Render startup script — writes Streamlit secrets from env vars before launch.

mkdir -p .streamlit

cat > .streamlit/secrets.toml <<EOF
ANTHROPIC_API_KEY = "${ANTHROPIC_API_KEY}"
CREDITS_DATABASE_URL = "${CREDITS_DATABASE_URL}"

[users.brandon]
password_hash = "${USER_BRANDON_HASH}"
role = "admin"
credits_per_week = 20
EOF

exec streamlit run app.py \
  --server.port "${PORT:-8501}" \
  --server.address 0.0.0.0 \
  --server.headless true \
  --browser.gatherUsageStats false
