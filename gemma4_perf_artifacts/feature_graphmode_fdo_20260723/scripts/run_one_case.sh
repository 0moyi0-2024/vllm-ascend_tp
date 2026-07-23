#!/usr/bin/env bash
# run_one_case.sh — run one feature-verification case end-to-end.
# Env vars (all required unless noted):
#   CASE_NAME, MODEL_PATH, SERVED_NAME, DEVICES (e.g. "0,1"), PORT,
#   BASE_FLAGS (model-specific flags string), FEATURE_FLAGS (differential flags string),
#   ADD_CONFIG (JSON for --additional-config; may be empty),
#   BENCH_INPUT, BENCH_OUTPUT, NUM_PROMPTS, REQUEST_RATE,
#   CASE_KIND (baseline|chunked|prefix|weightnz|async|cpubind|chunked_prefix),
#   OUT (root output dir)
set -uo pipefail

: "${CASE_NAME:?}"; : "${MODEL_PATH:?}"; : "${SERVED_NAME:?}"; : "${DEVICES:?}"; : "${PORT:?}"
: "${BASE_FLAGS:=}"; : "${FEATURE_FLAGS:=}"; : "${ADD_CONFIG:=}"
: "${BENCH_INPUT:=2048}"; : "${BENCH_OUTPUT:=128}"; : "${NUM_PROMPTS:=128}"; : "${REQUEST_RATE:=8}"
: "${CASE_KIND:=baseline}"; : "${OUT:?}"

CASE_DIR="${OUT}/cases/${CASE_NAME}"
mkdir -p "${CASE_DIR}"

CWD=/home/vllm-ascend_tp/vllm-ascend_tp   # vllm-ascend editable install path (required launch cwd)
LOG() { echo "[$(date '+%F %T')] [${CASE_NAME}] $*" | tee -a "${CASE_DIR}/runner.log"; }

# Recursively collect all descendant pids of a pid (while parent still alive;
# vllm EngineCore/Worker_TP escape the process group once the parent dies).
get_descendants() {
  local root=$1 all=" $root " changed=1 c cc
  while [ $changed -eq 1 ]; do
    changed=0
    for p in $all; do
      c=$(pgrep -P "$p" 2>/dev/null)
      [ -n "$c" ] || continue
      for cc in $c; do
        case "$all" in *" $cc "*) ;; *) all="$all$cc "; changed=1;; esac
      done
    done
  done
  echo "$all"
}

# Kill orphaned vllm worker processes (PPID 1 = parent already dead).
# Only targets reparented leftovers, never an active server (whose workers have a live parent).
kill_orphan_vllm_workers() {
  ps -eo pid,ppid,args 2>/dev/null | awk '$2==1 && ($0 ~ /VLLM::EngineCore|VLLM::Worker_TP/){print $1}' | while read p; do kill -9 "$p" 2>/dev/null; done
}

# Free assigned NPU cards: kill orphaned vllm workers + any pid npu-smi reports on these cards.
clear_cards() {
  local devs="$1" i pids
  kill_orphan_vllm_workers
  for i in $(seq 1 6); do
    pids=$(npu-smi info 2>/dev/null | python3 -c "
import sys,re
devs=set('${devs}'.split(','))
out=[]
for line in sys.stdin:
    m=re.match(r'\|\s+(\d+)\s+\d+\s+\|\s+(\d+)\s+\|', line)
    if m and m.group(1) in devs: out.append(m.group(2))
print(' '.join(out))
" 2>/dev/null)
    [ -z "$pids" ] && return 0
    for p in $pids; do kill -9 "$p" 2>/dev/null; done
    kill_orphan_vllm_workers
    sleep 3
  done
}

shutdown_server() {
  LOG "shutting down server (graceful TERM, then force descendants, then card clear)"
  local desc=""
  [ -n "${SERVER_PID:-}" ] && desc=$(get_descendants "${SERVER_PID}")
  kill -TERM "-${SERVER_PGID}" 2>/dev/null || true
  for i in $(seq 1 25); do kill -0 "${SERVER_PGID}" 2>/dev/null || break; sleep 1; done
  if [ -n "$desc" ]; then for p in $desc; do kill -9 "$p" 2>/dev/null; done; fi
  kill -KILL "-${SERVER_PGID}" 2>/dev/null || true
  sleep 2
  clear_cards "${DEVICES}"
}

start_ts=$(date +%s)
LOG "START case=${CASE_NAME} kind=${CASE_KIND} devices=${DEVICES} port=${PORT}"

# Pre-flight: ensure assigned cards are free (leftovers from a prior failed case)
LOG "pre-flight: clearing cards ${DEVICES}"
clear_cards "${DEVICES}"
sleep 3

# Build command
ADD_ARG=""
[ -n "${ADD_CONFIG}" ] && ADD_ARG="--additional-config '${ADD_CONFIG}'"
# COMMON_FLAGS: FULL_DECODE_ONLY graph mode at TP4 (TP4 splits head_dim to 128,
# avoiding the 512-dim PA fallback whose graph_task_group capture crashes with
# ACL 107033 at TP1/TP2). FULL_DECODE_ONLY is not usable at TP2 for gemma4.
#   --max-model-len 8192 : default 256k OOMs at profiling
#   --block-size 128 : Ascend attention backends only support [128] (default 16 -> "No common block size")
COMMON_FLAGS="--compilation-config '{\"cudagraph_mode\":\"FULL_DECODE_ONLY\"}' --max-model-len 8192 --block-size 128"
CMD="cd ${CWD} && ASCEND_RT_VISIBLE_DEVICES=${DEVICES} HCCL_OP_EXPANSION_MODE=AIV HCCL_BUFFSIZE=256 vllm serve ${MODEL_PATH} --served-model-name ${SERVED_NAME} --tensor-parallel-size 4 ${BASE_FLAGS} ${COMMON_FLAGS} ${FEATURE_FLAGS} ${ADD_ARG} --host 0.0.0.0 --port ${PORT}"
echo "${CMD}" > "${CASE_DIR}/command.sh"
chmod +x "${CASE_DIR}/command.sh"
LOG "command: ${CMD}"

# Start server in its own process group (setsid), detached
setsid bash -c "${CMD}" > "${CASE_DIR}/server.log" 2>&1 &
SERVER_PGID=$!
echo "${SERVER_PGID}" > "${CASE_DIR}/server.pgid"
# actual vllm pid (the bash -c child)
sleep 2
SERVER_PID=$(pgrep -f "vllm serve ${MODEL_PATH}.*--port ${PORT}" | head -n1)
echo "${SERVER_PID}" > "${CASE_DIR}/server.pid"
LOG "server pgid=${SERVER_PGID} pid=${SERVER_PID}"

# Wait for /v1/models healthy
ready=0
for i in $(seq 1 180); do   # up to 30 min
  if ! kill -0 "${SERVER_PGID}" 2>/dev/null; then
    LOG "ERROR: server process group died during boot"
    break
  fi
  resp=$(curl -s -m 5 "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null)
  if echo "${resp}" | grep -q "${SERVED_NAME}"; then
    ready=1; LOG "server ready after ${i}0s"
    break
  fi
  sleep 10
done

if [ "${ready}" != "1" ]; then
  LOG "FAIL: server not ready (boot timeout/crash)"
  echo "BOOT_FAIL" > "${CASE_DIR}/status.txt"
  echo "0" > "${CASE_DIR}/exitcode.txt"
  # save tail of log
  tail -100 "${CASE_DIR}/server.log" > "${CASE_DIR}/server_tail.log" 2>/dev/null
  shutdown_server
  exit 1
fi

echo "${SERVED_NAME}" > "${CASE_DIR}/models.txt"
curl -s "http://127.0.0.1:${PORT}/v1/models" > "${CASE_DIR}/models.json" 2>/dev/null

# Smoke test
cat > "${CASE_DIR}/smoke_payload.json" <<JSON
{
  "model": "${SERVED_NAME}",
  "messages": [{"role": "user", "content": "用一句话说明 vLLM 是什么。"}],
  "max_tokens": 64, "temperature": 0
}
JSON
smoke=$(curl -s -m 60 "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" -d @"${CASE_DIR}/smoke_payload.json" 2>/dev/null)
echo "${smoke}" > "${CASE_DIR}/smoke_response.json"
if echo "${smoke}" | grep -q '"choices"'; then
  LOG "smoke OK"
else
  LOG "WARN: smoke response missing choices"
fi

# Functional verification: network-free request loop (no vllm bench / no HF network).
# Drives N chat requests via urllib to localhost, records success count + latency.
LOG "functional request loop (12 requests)"
python3 - "${PORT}" "${SERVED_NAME}" "${CASE_DIR}/func_verify.json" <<'PY' > "${CASE_DIR}/func_verify.log" 2>&1
import json, time, urllib.request, sys, statistics
port, model, outpath = sys.argv[1], sys.argv[2], sys.argv[3]
url = f"http://127.0.0.1:{port}/v1/chat/completions"
prompts = ["What is 2+2?", "Name a prime number.", "Say hello in French.",
           "What color is the sky?", "Count to five.", "Capital of France?",
           "What is water made of?", "Name a mammal.", "2*3=?", "Define a noun.",
           "Largest planet?", "What is vLLM?"]
lat, ok = [], 0
for i, p in enumerate(prompts):
    payload = {"model": model, "messages":[{"role":"user","content":p}],
               "max_tokens": 32, "temperature": 0}
    t0 = time.time()
    try:
        req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                     headers={"Content-Type":"application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            body = json.loads(r.read())
        dt = time.time() - t0
        if "choices" in body:
            ok += 1; lat.append(dt)
            print(f"[{i}] ok dt={dt:.2f}s reply={body['choices'][0]['message']['content'][:40]!r}", flush=True)
        else:
            print(f"[{i}] no choices: {str(body)[:120]}", flush=True)
    except Exception as e:
        print(f"[{i}] FAIL {type(e).__name__}: {e}", flush=True)
    time.sleep(0.5)
summary = {"ok": ok, "total": len(prompts),
           "latency_mean_s": round(statistics.mean(lat),3) if lat else None,
           "latency_max_s": round(max(lat),3) if lat else None}
json.dump(summary, open(outpath,"w"), indent=2)
print("SUMMARY", json.dumps(summary), flush=True)
PY
func_rc=$?
LOG "functional loop done rc=${func_rc} ($(cat ${CASE_DIR}/func_verify.json 2>/dev/null))"
echo "${func_rc}" > "${CASE_DIR}/func_verify.exitcode"

# Config-effect evidence: prove the feature flag actually took effect in the engine config
LOG "extracting config-effect evidence"
{
  echo "===== ENGINE CONFIG (non-default args, feature-relevant) ====="
  grep -oE "non-default args: .*" "${CASE_DIR}/server.log" 2>/dev/null | head -1 | tr ',' '\n' | grep -iE "chunked_prefill|prefix_caching|async_scheduling|block_size|max_model_len|weight_nz|cpu_binding|additional_config|enforce_eager" || true
  echo "===== FEATURE-SPECIFIC EVIDENCE ====="
  case "${CASE_KIND}" in
    chunked) grep -iE "chunked prefill|max_num_batched_tokens|ChunkedPrefill" "${CASE_DIR}/server.log" 2>/dev/null | tail -5 ;;
    prefix)  grep -iE "prefix cach|enable_prefix_caching|Prefix cache hit" "${CASE_DIR}/server.log" 2>/dev/null | tail -8 ;;
    weightnz) grep -iE "weight_nz|nz_mode|enable_nz" "${CASE_DIR}/server.log" 2>/dev/null | tail -8 ;;
    async)   grep -iE "async schedul|AsyncScheduler|async_scheduling" "${CASE_DIR}/server.log" 2>/dev/null | tail -8 ;;
    cpubind) grep -iE "cpu binding|bind_cpus|enable_cpu_binding|affinity" "${CASE_DIR}/server.log" 2>/dev/null | tail -8 ;;
    chunked_prefix) grep -iE "chunked prefill|prefix cach" "${CASE_DIR}/server.log" 2>/dev/null | tail -8 ;;
  esac
} > "${CASE_DIR}/config_evidence.txt" 2>/dev/null

# Prefix-cache hit verification: shared-prefix workload (network-free) + metrics hit rate
if [ "${CASE_KIND}" = "prefix" ]; then
  LOG "running shared-prefix workload (APC hit verification)"
  python "${OUT}/scripts/shared_prefix_bench.py" --port "${PORT}" --model "${SERVED_NAME}" \
    --prefix-len 4096 --num-prompts 40 --request-rate 8 \
    > "${CASE_DIR}/bench_shared_prefix.log" 2>&1 || true
  curl -s "http://127.0.0.1:${PORT}/metrics" 2>/dev/null | grep -iE "prefix_cache|cache_hit|gpu_cache" \
    > "${CASE_DIR}/prefix_cache_metrics.txt" 2>/dev/null || true
fi

# Metrics + evidence
curl -s "http://127.0.0.1:${PORT}/metrics" > "${CASE_DIR}/metrics.prom" 2>/dev/null
cp "${CASE_DIR}/metrics.prom" "${OUT}/metrics/${CASE_NAME}.prom" 2>/dev/null
grep -Ei "chunk|prefix|cache|hit|async|scheduler|AsyncScheduler|nz|weight_nz|VLLM_ASCEND_ENABLE_NZ|cpu|binding|affinity|numa|migrate|irq|error|warn|exception|traceback" \
  "${CASE_DIR}/server.log" > "${CASE_DIR}/evidence_grep.txt" 2>/dev/null || true
grep -Ei "prefix|cache|hit|kv|ttft|tpot|itl|throughput|request|latency" \
  "${CASE_DIR}/metrics.prom" > "${CASE_DIR}/metrics_relevant.prom" 2>/dev/null || true

# CPU affinity capture (always; meaningful for cpubind case)
PID=$(pgrep -f "vllm serve ${MODEL_PATH}.*--port ${PORT}" | head -n1)
if [ -n "${PID}" ]; then
  taskset -cp "${PID}" > "${CASE_DIR}/taskset_main.txt" 2>/dev/null || true
  grep Cpus_allowed_list "/proc/${PID}/status" > "${CASE_DIR}/cpus_allowed_main.txt" 2>/dev/null || true
  ps -T -p "${PID}" -o pid,tid,psr,comm > "${CASE_DIR}/threads_psr.txt" 2>/dev/null || true
  for T in /proc/${PID}/task/*/status; do
    echo "### $T"; grep -E "Name|Cpus_allowed_list|Mems_allowed_list" "$T" 2>/dev/null
  done > "${CASE_DIR}/thread_affinity.txt" 2>/dev/null || true
fi

# Shutdown server
shutdown_server

end_ts=$(date +%s)
echo "OK" > "${CASE_DIR}/status.txt"
echo "0" > "${CASE_DIR}/exitcode.txt"
[ "${func_rc}" != "0" ] && echo "FUNC_FAIL" > "${CASE_DIR}/status.txt"
LOG "END case=${CASE_NAME} status=$(cat ${CASE_DIR}/status.txt) duration=$((end_ts-start_ts))s"
exit 0
