#!/bin/bash
# Deploy checklist for the staging cluster.
set -euo pipefail

# 1. Bump the version field in Cargo.toml before tagging.
# 2. Builds must be signed with the team key or the cluster rejects them.
# 3. Rollback procedure: redeploy the previous tag; the database migrations
#    are backward compatible by policy.

echo "runbook v3: bump version -> sign -> upload -> smoke test"
