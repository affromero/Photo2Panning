#!/usr/bin/env bash
set -euo pipefail

violations=$(
    git ls-files | awk '
        $0 ~ /^(dependencies|experiments)\// { next }
        $0 ~ /^tests\/.+\/tests(\/|$)/ { print; next }
        $0 ~ /^tests\// { next }
        $0 ~ /(^|\/)tests(\/|$)/ { print; next }
        $0 ~ /(^|\/)test_[^\/]*\.py$/ { print; next }
        $0 ~ /(^|\/)[^\/]*_test\.py$/ { print; next }
    '
)

if [ -n "$violations" ]; then
    echo "Test files and test folders must live under root tests/ without nested tests/ directories." >&2
    echo "$violations" >&2
    exit 1
fi
