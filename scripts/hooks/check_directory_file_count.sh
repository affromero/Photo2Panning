#!/usr/bin/env bash
set -euo pipefail

max_files="${PHOTO2PANNING_MAX_FILES_PER_DIR:-10}"
include_regex=""
exclude_regex=""

while [ "$#" -gt 0 ]; do
    case "$1" in
        --include-regex)
            shift
            if [ "$#" -eq 0 ]; then
                echo "Missing value for --include-regex." >&2
                exit 2
            fi
            if [ -n "$include_regex" ]; then
                include_regex="${include_regex}|($1)"
            else
                include_regex="($1)"
            fi
            ;;
        --include-regex=*)
            value="${1#--include-regex=}"
            if [ -n "$include_regex" ]; then
                include_regex="${include_regex}|(${value})"
            else
                include_regex="(${value})"
            fi
            ;;
        --exclude-regex)
            shift
            if [ "$#" -eq 0 ]; then
                echo "Missing value for --exclude-regex." >&2
                exit 2
            fi
            if [ -n "$exclude_regex" ]; then
                exclude_regex="${exclude_regex}|($1)"
            else
                exclude_regex="($1)"
            fi
            ;;
        --exclude-regex=*)
            value="${1#--exclude-regex=}"
            if [ -n "$exclude_regex" ]; then
                exclude_regex="${exclude_regex}|(${value})"
            else
                exclude_regex="(${value})"
            fi
            ;;
        *)
            echo "Unexpected argument: $1" >&2
            exit 2
            ;;
    esac
    shift
done

violations=$(
    git ls-files | awk \
        -v max_files="$max_files" \
        -v include_regex="$include_regex" \
        -v exclude_regex="$exclude_regex" '
        $0 == "" { next }
        include_regex != "" && $0 !~ include_regex { next }
        exclude_regex != "" && $0 ~ exclude_regex { next }
        {
            dir = "."
            if ($0 ~ /\//) {
                dir = $0
                sub("/[^/]*$", "", dir)
            }
            counts[dir] += 1
        }
        END {
            for (dir in counts) {
                if (counts[dir] > max_files) {
                    printf "%s: %d files\n", dir, counts[dir]
                }
            }
        }
    ' | sort
)

if [ -n "$violations" ]; then
    echo "Directories may contain at most ${max_files} matching tracked files." >&2
    echo "Split broad folders into focused subdirectories:" >&2
    echo "$violations" >&2
    exit 1
fi
