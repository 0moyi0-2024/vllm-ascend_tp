#!/usr/bin/env bash
set -u
OUT=$(cat /tmp/graphmode_out_path.txt)
SUMMARY="$OUT/reports/case_summary.tsv"
mkdir -p "$OUT/reports"
echo -e "case\tstatus\tfunc_ok\tfunc_total\tlat_mean_s\tlat_max_s\tconfig_evidence_lines\texitcode" > "$SUMMARY"

for d in "$OUT"/cases/*/; do
  [ -d "$d" ] || continue
  name=$(basename "$d")
  status="unknown"; [ -f "$d/status.txt" ] && status=$(cat "$d/status.txt")
  rc="-"; [ -f "$OUT/status/${name}.exitcode" ] && rc=$(cat "$OUT/status/${name}.exitcode")
  ok="-"; tot="-"; lm="-"; lmax="-"; ev="-"
  if [ -f "$d/func_verify.json" ]; then
    ok=$(python3 -c "import json;print(json.load(open('$d/func_verify.json'))['ok'])" 2>/dev/null)
    tot=$(python3 -c "import json;print(json.load(open('$d/func_verify.json'))['total'])" 2>/dev/null)
    lm=$(python3 -c "import json;print(json.load(open('$d/func_verify.json'))['latency_mean_s'])" 2>/dev/null)
    lmax=$(python3 -c "import json;print(json.load(open('$d/func_verify.json'))['latency_max_s'])" 2>/dev/null)
  fi
  [ -f "$d/config_evidence.txt" ] && ev=$(grep -cve '^\s*$' "$d/config_evidence.txt" 2>/dev/null)
  echo -e "${name}\t${status}\t${ok}\t${tot}\t${lm}\t${lmax}\t${ev}\t${rc}" >> "$SUMMARY"
done
echo "Wrote $SUMMARY"
column -t -s$'\t' "$SUMMARY"
