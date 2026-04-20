#!/usr/bin/env bash

# Optional shell exports for local client-side development.
# The primary SkillClaw client config lives in ~/.skillclaw/config.yaml.
#
# Typical usage:
#   cp client_env.example.sh client_env.local.sh
#   source client_env.local.sh
#
# These values are environment fallbacks and teammate-friendly examples.
# They are not the evolve server's `EVOLVE_STORAGE_*` settings.

export OPENAI_BASE_URL="https://your-model-gateway.example/v1"
export OPENAI_API_KEY="your-openai-api-key"

# Convenience values for shared-skill setup discussions or local wrappers.
export OSS_ENDPOINT="https://oss-cn-hangzhou.aliyuncs.com"
export OSS_BUCKET="skillclaw-shared-skills"
export OSS_ACCESS_KEY_ID="your-oss-access-key-id"
export OSS_ACCESS_KEY_SECRET="your-oss-access-key-secret"
export OSS_GROUP_ID="your-group-id"
export OSS_USER_ALIAS="your-user-alias"
