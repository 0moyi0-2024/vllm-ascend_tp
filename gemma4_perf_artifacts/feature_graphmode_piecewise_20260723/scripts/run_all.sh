#!/usr/bin/env bash
set -uo pipefail
OUT=$(cat /tmp/graphmode_out_path.txt)
export OUT
CWD=/home/vllm-ascend_tp/vllm-ascend_tp
cd "${CWD}"

mkdir -p "${OUT}/logs" "${OUT}/pids" "${OUT}/status"

# ---- Model base flags ----
BASE_31B="--enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 --limit-mm-per-prompt '{\"image\":2,\"audio\":1,\"video\":1}'"
BASE_26B="--enable-expert-parallel --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 --limit-mm-per-prompt '{\"image\":2,\"audio\":1,\"video\":1}'"

# ---- Feature differentials: KIND -> "FEATURE_FLAGS|ADD_CONFIG" ----
cfg_baseline='--no-enable-prefix-caching --no-enable-chunked-prefill --no-async-scheduling|{"enable_cpu_binding": false, "weight_nz_mode": 0}'
cfg_chunked='--no-enable-prefix-caching --enable-chunked-prefill --max-num-batched-tokens 2048 --no-async-scheduling|{"enable_cpu_binding": false, "weight_nz_mode": 0}'
cfg_prefix='--enable-prefix-caching --no-enable-chunked-prefill --no-async-scheduling|{"enable_cpu_binding": false, "weight_nz_mode": 0}'
cfg_weightnz='--no-enable-prefix-caching --no-enable-chunked-prefill --no-async-scheduling|{"enable_cpu_binding": false, "weight_nz_mode": 2}'
cfg_async='--no-enable-prefix-caching --no-enable-chunked-prefill --async-scheduling|{"enable_cpu_binding": false, "weight_nz_mode": 0}'
cfg_cpubind='--no-enable-prefix-caching --no-enable-chunked-prefill --no-async-scheduling|{"enable_cpu_binding": true, "weight_nz_mode": 0}'
cfg_chunked_prefix='--enable-prefix-caching --enable-chunked-prefill --max-num-batched-tokens 2048 --no-async-scheduling|{"enable_cpu_binding": false, "weight_nz_mode": 0}'

# ---- Case list: NAME|MODEL_PATH|SERVED|BASE_VAR|KIND ----
CASES=(
  "31B_baseline|/home/xty/gemma4/31B|gemma-4-31B-it|BASE_31B|baseline"
  "31B_chunked|/home/xty/gemma4/31B|gemma-4-31B-it|BASE_31B|chunked"
  "31B_prefix|/home/xty/gemma4/31B|gemma-4-31B-it|BASE_31B|prefix"
  "31B_weightnz|/home/xty/gemma4/31B|gemma-4-31B-it|BASE_31B|weightnz"
  "31B_async|/home/xty/gemma4/31B|gemma-4-31B-it|BASE_31B|async"
  "31B_cpubind|/home/xty/gemma4/31B|gemma-4-31B-it|BASE_31B|cpubind"
  "31B_chunked_prefix|/home/xty/gemma4/31B|gemma-4-31B-it|BASE_31B|chunked_prefix"
  "26B_baseline|/home/xty/gemma4/26B|gemma-4-26B-A4B-it|BASE_26B|baseline"
  "26B_chunked|/home/xty/gemma4/26B|gemma-4-26B-A4B-it|BASE_26B|chunked"
  "26B_prefix|/home/xty/gemma4/26B|gemma-4-26B-A4B-it|BASE_26B|prefix"
  "26B_weightnz|/home/xty/gemma4/26B|gemma-4-26B-A4B-it|BASE_26B|weightnz"
  "26B_async|/home/xty/gemma4/26B|gemma-4-26B-A4B-it|BASE_26B|async"
  "26B_cpubind|/home/xty/gemma4/26B|gemma-4-26B-A4B-it|BASE_26B|cpubind"
  "26B_chunked_prefix|/home/xty/gemma4/26B|gemma-4-26B-A4B-it|BASE_26B|chunked_prefix"
)

# 4 slots: each owns 2 NPUs + a port. Cases round-robin assigned.
SLOT_DEVS=("0,1" "2,3" "4,5" "6,7")
SLOT_PORTS=("9000" "9002" "9004" "9006")

# Round-robin assign cases to slots
declare -a SLOT_CASES_0 SLOT_CASES_1 SLOT_CASES_2 SLOT_CASES_3
for i in "${!CASES[@]}"; do
  slot=$((i % 4))
  eval "SLOT_CASES_${slot}+=(\"\${CASES[\$i]}\")"
done

run_slot() {
  local slot=$1
  local devs="${SLOT_DEVS[$slot]}"
  local port="${SLOT_PORTS[$slot]}"
  local -a arr
  eval "arr=(\"\${SLOT_CASES_${slot}[@]}\")"
  for spec in "${arr[@]}"; do
    IFS='|' read -r NAME MODEL_PATH SERVED BASE_VAR KIND <<< "$spec"
    local base_flags
    eval "base_flags=\"\${${BASE_VAR}}\""
    local cfg_var="cfg_${KIND}"
    local cfg
    eval "cfg=\"\${${cfg_var}}\""
    local feature_flags="${cfg%%|*}"
    local add_config="${cfg##*|}"
    echo "[$(date '+%F %T')] slot${slot} START ${NAME} (devs=${devs} port=${port})" >> "${OUT}/logs/run_all.log"
    CASE_NAME="${NAME}" MODEL_PATH="${MODEL_PATH}" SERVED_NAME="${SERVED}" \
      DEVICES="${devs}" PORT="${port}" BASE_FLAGS="${base_flags}" \
      FEATURE_FLAGS="${feature_flags}" ADD_CONFIG="${add_config}" \
      BENCH_INPUT=2048 BENCH_OUTPUT=128 NUM_PROMPTS=128 REQUEST_RATE=8 \
      CASE_KIND="${KIND}" OUT="${OUT}" \
      bash "${OUT}/scripts/run_one_case.sh" >> "${OUT}/logs/slot${slot}_${NAME}.log" 2>&1
    rc=$?
    echo "${rc}" > "${OUT}/status/${NAME}.exitcode"
    echo "[$(date '+%F %T')] slot${slot} END ${NAME} rc=${rc}" >> "${OUT}/logs/run_all.log"
    # cooldown between cases on same cards
    sleep 15
  done
}

export -f run_slot
export SLOT_DEVS SLOT_PORTS BASE_31B BASE_26B
export cfg_baseline cfg_chunked cfg_prefix cfg_weightnz cfg_async cfg_cpubind cfg_chunked_prefix
export OUT

# Pre-flight: ensure no stray vllm processes holding cards
pkill -9 -f "vllm serve /home/xty/gemma4" 2>/dev/null || true
sleep 5

echo "[$(date '+%F %T')] run_all START, ${#CASES[@]} cases, 4 parallel slots" | tee -a "${OUT}/logs/run_all.log"

# Launch 4 slot workers in background
run_slot 0 >> "${OUT}/logs/slot0.log" 2>&1 &
echo $! > "${OUT}/pids/slot0.pid"
run_slot 1 >> "${OUT}/logs/slot1.log" 2>&1 &
echo $! > "${OUT}/pids/slot1.pid"
run_slot 2 >> "${OUT}/logs/slot2.log" 2>&1 &
echo $! > "${OUT}/pids/slot2.pid"
run_slot 3 >> "${OUT}/logs/slot3.log" 2>&1 &
echo $! > "${OUT}/pids/slot3.pid"

# Wait for all slots
wait
echo "[$(date '+%F %T')] run_all ALL SLOTS DONE" | tee -a "${OUT}/logs/run_all.log"

bash "${OUT}/scripts/collect_results.sh" >> "${OUT}/logs/collect.log" 2>&1 || true
echo "[$(date '+%F %T')] run_all COMPLETE" | tee -a "${OUT}/logs/run_all.log"
