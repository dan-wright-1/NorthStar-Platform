#!/usr/bin/env bash
# data-bucket.sh
# Keep the S3 data bucket (and the Lab 2 data in it) across teardowns.
#
# The data bucket is created by module.storage in the dev stack, so a plain
# `terraform destroy` deletes it and everything in it (force_destroy = true).
# S3 storage is cheap and rebuilding the data means re-running the Glue jobs, so
# instead of destroying the bucket we detach it from Terraform state first:
# destroy then no longer knows it exists and leaves it alone.
#
#   detach  - before destroy: remove module.storage.* from state. The bucket and
#             its objects stay in S3.
#   attach  - before apply on a rebuilt stack: import the surviving bucket back
#             into state so apply does not fail with BucketAlreadyOwnedByYou.
#             Does nothing when the bucket is not there (first-time build).
#
# Usage: bash scripts/data-bucket.sh detach|attach
# Needs AWS_PROFILE=tf (credentials that refresh) and `terraform init` already run.

set -uo pipefail
export MSYS_NO_PATHCONV=1

PROJECT="${PROJECT:-northstar}"
ENVIRONMENT="${ENVIRONMENT:-dev}"
TF_DIR="infrastructure/environments/${ENVIRONMENT}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text 2>/dev/null | tr -d '\r')
if [ -z "${ACCOUNT_ID}" ]; then
  echo "==> ERROR: AWS CLI is not authenticated. Nothing was changed."; exit 1
fi
BUCKET="${PROJECT}-${ENVIRONMENT}-data-${ACCOUNT_ID}"
tf () { terraform -chdir="${TF_DIR}" "$@"; }

case "${1:-}" in
  detach)
    # List what is in state rather than hardcoding addresses, so a module change
    # does not leave one resource behind to be destroyed (and fail on BucketNotEmpty).
    addrs=$(tf state list 2>/dev/null | tr -d '\r' | grep '^module\.storage\.')
    if [ -z "${addrs}" ]; then
      echo "==> No module.storage resources in state - nothing to detach."; exit 0
    fi
    echo "==> Detaching ${BUCKET} from Terraform state (data stays in S3):"
    echo "${addrs}" | sed 's/^/      /'
    while IFS= read -r a; do
      tf state rm "${a}" >/dev/null || { echo "==> ERROR removing ${a}"; exit 1; }
    done <<< "${addrs}"
    echo "==> Done. terraform destroy will now leave the bucket alone."
    ;;
  attach)
    if ! aws s3api head-bucket --bucket "${BUCKET}" 2>/dev/null; then
      echo "==> ${BUCKET} does not exist - nothing to import (first-time build)."; exit 0
    fi
    if tf state list 2>/dev/null | tr -d '\r' | grep -q '^module\.storage\.aws_s3_bucket\.data$'; then
      echo "==> ${BUCKET} is already in state."; exit 0
    fi
    echo "==> Importing ${BUCKET} into Terraform state"
    imp () { echo "      $1"; tf import -input=false "$1" "$2" >/dev/null || { echo "==> ERROR importing $1"; exit 1; }; }
    imp 'module.storage.aws_s3_bucket.data'                                    "${BUCKET}"
    imp 'module.storage.aws_s3_bucket_versioning.data'                         "${BUCKET}"
    imp 'module.storage.aws_s3_bucket_public_access_block.data'                "${BUCKET}"
    imp 'module.storage.aws_s3_bucket_server_side_encryption_configuration.data' "${BUCKET}"
    imp 'module.storage.aws_s3_bucket_lifecycle_configuration.data[0]'         "${BUCKET}"
    for p in artifacts features processed raw; do
      imp "module.storage.aws_s3_object.prefixes[\"${p}/\"]" "${BUCKET}/${p}/"
    done
    echo "==> Done. Run 'terraform plan' - the storage module should show no changes."
    ;;
  *)
    echo "Usage: bash scripts/data-bucket.sh detach|attach"; exit 2 ;;
esac
