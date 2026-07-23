#!/usr/bin/env bash
set -u
OUT=$(cat /tmp/fdo_out_path.txt 2>/dev/null || echo "/home/xty/gemma4_vllm_ascend_feature_verify_latest")
echo "OUT=$OUT"
echo "==== run_all proc ===="
ps -ef | grep -E "run_all|run_one_case|vllm serve /home/xty/gemma4" | grep -v grep || echo "(none running)"
echo "==== slot pids ===="
for s in 0 1 2 3; do
  f="$OUT/pids/slot${s}.pid"
  [ -f "$f" ] && { p=$(cat "$f"); echo "slot${s} pid=$p alive=$(kill -0 $p 2>/dev/null && echo yes || echo no)"; }
done
echo "==== case status ===="
for c in $OUT/cases/*/status.txt; do [ -f "$c" ] && echo "$(dirname $c | xargs basename): $(cat $c)"; done 2>/dev/null
echo "==== exit codes ===="
for c in $OUT/status/*.exitcode; do [ -f "$c" ] && echo "$(basename $c .exitcode): $(cat $c)"; done 2>/dev/null
echo "==== run_all log tail ===="
tail -20 "$OUT/logs/run_all.log" 2>/dev/null || true
echo "==== recent errors in server logs ===="
grep -RniE "error|exception|traceback|failed|oom|out of memory" "$OUT/cases"/*/server.log 2>/dev/null | tail -30 || echo "(none)"
