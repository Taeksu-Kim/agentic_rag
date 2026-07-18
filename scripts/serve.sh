#!/usr/bin/env bash
# agentic_rag 서빙 스택 기동/정지 런북 (RTX 4090 24GB, WSL2)
#
#   bash scripts/serve.sh start   # 전부 순서대로 기동 (이미 떠 있는 건 건너뜀)
#   bash scripts/serve.sh stop    # 전부 정지 (좀비 EngineCore 포함)
#   bash scripts/serve.sh status  # 상태 확인
#
# 구성 (docs/design_and_plan.md §2 — 검증된 최종 형태):
#   :6333  Qdrant (native, CPU)
#   :8011  임베더 — ★CPU★ (transformers, vllm env python). GPU 임베더와 코사인
#          0.9998+ 일치 검증됨. 이 vLLM 버전은 free>=util*total 사전검사 +
#          타 프로세스 포함 KV 회계 때문에 24GB에서 GPU 3모델 동시 상주가 불가
#          → 임베더를 CPU로 내려 GPU를 9B+리랭커에 양보.
#   :8002  Qwen3-Reranker-0.6B (GPU, util 0.12) — 9B보다 먼저 띄울 것
#   :8000  Qwen3.5-9B (GPU, util은 아래 NINEB_UTIL) — 반드시 마지막
#
# NINEB_UTIL 가이드 (util = "타 프로세스 포함 총 GPU 사용 상한"이라는 게 함정):
#   0.72  넷플릭스/브라우저 등 윈도우측 GPU 사용(~2-3GB)과 공존하는 안전값 (기본)
#   0.80  윈도우측이 조용할 때 (KV 여유 최대) — 벤치마크 장시간 실행 권장값
#
# 실험 모드: EMBED_GPU=1 bash scripts/serve.sh start
#   임베더도 GPU에 (3모델 동주). --kv-cache-memory-bytes로 KV를 명시해
#   프로파일링 회계를 우회하는 방식 — 물리 합 ~22GB로 들어가는 건 확인됐고
#   플래그 우회가 실제로 검사를 통과하는지는 첫 실행에서 검증 필요.
#   성공 시 .env EMBEDDER_URL을 :8001/v1 로 바꿔줄 것 (기본은 CPU :8011/v1).
# 윈도우측 점유는 WSL nvidia-smi에 프로세스명이 안 보이지만 총량엔 포함된다.
# 총 사용량이 23GB를 넘보면 WDDM이 시스템 RAM으로 스필 → 생성 0.5 tok/s 붕괴.
#
# 주의:
#  - 기동 순서 고정: Qdrant → CPU임베더 → 리랭커 → 9B(마지막). 동시 기동 금지
#    (프로파일링 경합 "Initial free memory ..." 어서션).
#  - stop은 'VLLM::EngineCore'도 죽인다 — API 서버만 죽이면 EngineCore가
#    좀비로 남아 4-5GB를 물고 다음 부팅을 망친다 (실측).
#  - 9B는 로컬 경로 필수 (HF repo id는 프로파일링 데드락).

set -u
VENV=$HOME/miniconda3/envs/vllm
CU=$VENV/lib/python3.12/site-packages/nvidia/cu13
export CUDA_HOME=$CU PATH=$CU/bin:$PATH LD_LIBRARY_PATH=$CU/lib:${LD_LIBRARY_PATH:-}
export VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONUNBUFFERED=1

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOGDIR=${LOGDIR:-$HOME/agentic_rag_logs}; mkdir -p "$LOGDIR"
NINEB_UTIL=${NINEB_UTIL:-0.72}
NINEB_MODEL="/mnt/d/workspace/stock dataset/models/Qwen3.5-9B-FP8-dynamic"
WSL_IP=${WSL_IP:-10.5.0.2}

up()   { curl -s --max-time 2 "$1" 2>/dev/null | grep -q "$2"; }
wait_up() { # url pattern logfile name
  local n=0
  until up "$1" "$2"; do
    if [ -f "$3" ] && grep -q "EngineCore failed" "$3"; then echo "!! $4 부팅 실패 — $3 확인"; return 1; fi
    sleep 5; n=$((n+5)); [ $n -ge 600 ] && { echo "!! $4 타임아웃"; return 1; }
  done
  echo "ok: $4"
}

start() {
  echo "== GPU: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader) (윈도우측 점유 포함)"

  # ① Qdrant
  if ! up http://localhost:6333/readyz ready; then
    QDRANT__STORAGE__STORAGE_PATH=$HOME/qdrant_storage QDRANT__SERVICE__HOST=0.0.0.0 \
      setsid nohup $HOME/qdrant/qdrant > "$LOGDIR/qdrant.log" 2>&1 &
    wait_up http://localhost:6333/readyz ready "$LOGDIR/qdrant.log" Qdrant || return 1
  else echo "ok: Qdrant (이미 실행 중)"; fi

  # ② 임베더 — 기본 CPU(:8011); EMBED_GPU=1이면 GPU(:8001, KV 명시 우회)
  if [ "${EMBED_GPU:-0}" = "1" ]; then
    if ! up "http://$WSL_IP:8001/v1/models" qwen3-emb; then
      setsid nohup $VENV/bin/vllm serve Qwen/Qwen3-Embedding-0.6B --runner pooling \
        --served-model-name qwen3-emb --max-model-len 4096 \
        --gpu-memory-utilization 0.10 --kv-cache-memory-bytes 536870912 --port 8001 \
        > "$LOGDIR/embedder.log" 2>&1 &
      wait_up "http://$WSL_IP:8001/v1/models" qwen3-emb "$LOGDIR/embedder.log" "GPU임베더" || return 1
    else echo "ok: GPU임베더 (이미 실행 중)"; fi
  else
    if ! up "http://$WSL_IP:8011/" qwen3-emb-cpu; then
      CUDA_VISIBLE_DEVICES="" setsid nohup $VENV/bin/python "$ROOT/scripts/cpu_embed_server.py" --port 8011 \
        > "$LOGDIR/cpu_embed.log" 2>&1 &
      wait_up "http://$WSL_IP:8011/" qwen3-emb-cpu "$LOGDIR/cpu_embed.log" "CPU임베더" || return 1
    else echo "ok: CPU임베더 (이미 실행 중)"; fi
  fi

  # ③ 리랭커 (GPU, 9B보다 먼저)
  if ! up "http://$WSL_IP:8002/v1/models" qwen3-reranker; then
    setsid nohup $VENV/bin/vllm serve Qwen/Qwen3-Reranker-0.6B --runner pooling \
      --served-model-name qwen3-reranker --max-model-len 2048 \
      --gpu-memory-utilization 0.12 --port 8002 \
      --hf-overrides '{"architectures":["Qwen3ForSequenceClassification"],"classifier_from_token":["no","yes"],"is_original_qwen3_reranker":true}' \
      > "$LOGDIR/reranker.log" 2>&1 &
    wait_up "http://$WSL_IP:8002/v1/models" qwen3-reranker "$LOGDIR/reranker.log" 리랭커 || return 1
  else echo "ok: 리랭커 (이미 실행 중)"; fi

  # ④ 9B (반드시 마지막). EMBED_GPU 모드에선 KV 명시로 회계 우회(1.5GiB).
  NINEB_EXTRA=""
  [ "${EMBED_GPU:-0}" = "1" ] && NINEB_EXTRA="--kv-cache-memory-bytes 1610612736"
  if ! up "http://$WSL_IP:8000/v1/models" qwen35-9b; then
    setsid nohup $VENV/bin/vllm serve "$NINEB_MODEL" \
      --served-model-name qwen35-9b --max-model-len 8192 --enforce-eager \
      --gpu-memory-utilization "$NINEB_UTIL" $NINEB_EXTRA --port 8000 \
      > "$LOGDIR/llm9b.log" 2>&1 &
    wait_up "http://$WSL_IP:8000/v1/models" qwen35-9b "$LOGDIR/llm9b.log" "9B" || return 1
    grep -E "KV cache memory" "$LOGDIR/llm9b.log" | tail -1
  else echo "ok: 9B (이미 실행 중)"; fi

  echo "== 완료. GPU: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader)"
}

stop() {
  pkill -f '[v]llm serve' 2>/dev/null
  pkill -9 -f 'VLLM::EngineCore' 2>/dev/null   # 좀비 방지 (필수)
  pkill -f '[c]pu_embed_server' 2>/dev/null
  sleep 4
  echo "정지 완료. GPU: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader) (남은 건 윈도우측)"
  echo "(Qdrant는 유지 — 내리려면: pkill -f qdrant/qdrant)"
}

status() {
  for x in "Qdrant|http://localhost:6333/readyz|ready" \
           "CPU임베더|http://$WSL_IP:8011/|qwen3-emb-cpu" \
           "리랭커|http://$WSL_IP:8002/v1/models|qwen3-reranker" \
           "9B|http://$WSL_IP:8000/v1/models|qwen35-9b"; do
    IFS='|' read -r name url pat <<< "$x"
    up "$url" "$pat" && echo "up:   $name" || echo "down: $name"
  done
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
}

case "${1:-start}" in
  start) start ;;
  stop) stop ;;
  status) status ;;
  *) echo "usage: serve.sh start|stop|status"; exit 1 ;;
esac
