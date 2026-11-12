#!/usr/bin/env bash
set -euo pipefail

: "${GH_TOKEN:?GH_TOKEN is required}"
: "${GH_REPO:?GH_REPO is required}"
: "${RELEASE_TAG:?RELEASE_TAG is required}"
: "${UBUNTU_X64_JOB_RESULT:?UBUNTU_X64_JOB_RESULT is required}"
: "${UBUNTU_ARM64_JOB_RESULT:?UBUNTU_ARM64_JOB_RESULT is required}"
: "${WINDOWS_X64_JOB_RESULT:?WINDOWS_X64_JOB_RESULT is required}"
: "${WINDOWS_ARM64_JOB_RESULT:?WINDOWS_ARM64_JOB_RESULT is required}"

download_missing_junit() {
  local junit_file="$1"
  local asset_name="$2"
  local destination

  if [[ -s "$junit_file" ]]; then
    return 0
  fi

  destination="$(dirname "$junit_file")"
  mkdir -p "$destination"
  gh release download "$RELEASE_TAG" \
    --pattern "$asset_name" \
    --dir "$destination" \
    --clobber >/dev/null 2>&1 || true
}

download_missing_junit "test-results/ubuntu-x64/test-${RELEASE_TAG}-ubuntu-x64-cpu.xml" "test-${RELEASE_TAG}-ubuntu-x64-cpu.xml"
download_missing_junit "test-results/ubuntu-arm64/test-${RELEASE_TAG}-ubuntu-arm64-cpu.xml" "test-${RELEASE_TAG}-ubuntu-arm64-cpu.xml"
download_missing_junit "test-results/windows-x64/test-${RELEASE_TAG}-windows-x64-cpu.xml" "test-${RELEASE_TAG}-windows-x64-cpu.xml"
download_missing_junit "test-results/windows-arm64/test-${RELEASE_TAG}-windows-arm64-cpu.xml" "test-${RELEASE_TAG}-windows-arm64-cpu.xml"

status_from_junit() {
  local junit_file="$1"
  local job_result="$2"

  if [[ ! -s "$junit_file" ]]; then
    printf 'JUnit XML unavailable (job result: %s)' "$job_result"
    return 0
  fi

  python3 - "$junit_file" <<'PYTHON'
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

path = Path(sys.argv[1])
try:
    root = ET.parse(path).getroot()
except (ET.ParseError, OSError) as exc:
    print(f"Invalid JUnit XML ({exc.__class__.__name__})")
    raise SystemExit(0)

def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]

suites = [node for node in root.iter() if local_name(node.tag) == "testsuite"]
if not suites and local_name(root.tag) == "testsuite":
    suites = [root]

leaf_suites = [
    suite for suite in suites
    if not any(local_name(child.tag) == "testsuite" for child in list(suite))
]
if leaf_suites:
    suites = leaf_suites

if not suites:
    print("Invalid JUnit XML (no testsuite elements)")
    raise SystemExit(0)

def count(suite: ET.Element, attribute: str, fallback_tag: str | None = None) -> int:
    value = suite.attrib.get(attribute)
    if value is not None:
        try:
            return int(float(value))
        except ValueError:
            pass
    if fallback_tag is not None:
        return sum(
            1
            for testcase in suite.iter()
            if local_name(testcase.tag) == "testcase"
            for child in list(testcase)
            if local_name(child.tag) == fallback_tag
        )
    return 0

tests = sum(count(suite, "tests") for suite in suites)
failures = sum(count(suite, "failures", "failure") for suite in suites)
errors = sum(count(suite, "errors", "error") for suite in suites)
skipped = sum(count(suite, "skipped", "skipped") for suite in suites)
failed = failures + errors
executed = max(tests - skipped, 0)
passed = max(executed - failed, 0)
percent = 100 if executed == 0 else int((passed * 100 / executed) + 0.5)
result = f"{percent}% tests passed, {failed} tests failed out of {tests}"
if skipped:
    result += f" ({skipped} skipped)"
print(result)
PYTHON
}

UBUNTU_X64_STATUS="$(status_from_junit "test-results/ubuntu-x64/test-${RELEASE_TAG}-ubuntu-x64-cpu.xml" "$UBUNTU_X64_JOB_RESULT")"
UBUNTU_ARM64_STATUS="$(status_from_junit "test-results/ubuntu-arm64/test-${RELEASE_TAG}-ubuntu-arm64-cpu.xml" "$UBUNTU_ARM64_JOB_RESULT")"
WINDOWS_X64_STATUS="$(status_from_junit "test-results/windows-x64/test-${RELEASE_TAG}-windows-x64-cpu.xml" "$WINDOWS_X64_JOB_RESULT")"
WINDOWS_ARM64_STATUS="$(status_from_junit "test-results/windows-arm64/test-${RELEASE_TAG}-windows-arm64-cpu.xml" "$WINDOWS_ARM64_JOB_RESULT")"

gh release view "$RELEASE_TAG" --json body --jq .body > release-notes-current.md
awk \
  -v ubuntu_x64="$UBUNTU_X64_STATUS" \
  -v ubuntu_arm64="$UBUNTU_ARM64_STATUS" \
  -v windows_x64="$WINDOWS_X64_STATUS" \
  -v windows_arm64="$WINDOWS_ARM64_STATUS" '
  /^Ubuntu x64 CPU: / {
    print "Ubuntu x64 CPU: " ubuntu_x64
    seen_ubuntu_x64 = 1
    next
  }
  /^Ubuntu ARM64 CPU: / {
    print "Ubuntu ARM64 CPU: " ubuntu_arm64
    seen_ubuntu_arm64 = 1
    next
  }
  /^Windows x64 CPU: / {
    print "Windows x64 CPU: " windows_x64
    seen_windows_x64 = 1
    next
  }
  /^Windows ARM64 CPU: / {
    print "Windows ARM64 CPU: " windows_arm64
    seen_windows_arm64 = 1
    next
  }
  { print }
  END {
    if (!seen_ubuntu_x64) print "Ubuntu x64 CPU: " ubuntu_x64
    if (!seen_ubuntu_arm64) print "Ubuntu ARM64 CPU: " ubuntu_arm64
    if (!seen_windows_x64) print "Windows x64 CPU: " windows_x64
    if (!seen_windows_arm64) print "Windows ARM64 CPU: " windows_arm64
  }
' release-notes-current.md > release-notes-updated.md

gh release edit "$RELEASE_TAG" --notes-file release-notes-updated.md
cat release-notes-updated.md
