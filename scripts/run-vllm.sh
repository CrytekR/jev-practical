#!/usr/bin/env bash
# 用官方镜像启动一个本地 vLLM OpenAI 服务。
#
# 镜像是 Docker Hub 上的 vllm/vllm-openai:v0.26.0，入口为 vllm serve。
# 0.26.0 的 Python 前端在 /v1/completions 上提供 logprob_token_ids，Jev 读出依赖它。
# 文档：https://docs.vllm.ai/en/v0.26.0/deployment/docker/
#
# 用法：
#   scripts/run-vllm.sh baselabs/mammoth-1 8001
#   scripts/run-vllm.sh Qwen/Qwen3.5-27B 8000
#   HF_TOKEN=hf_... scripts/run-vllm.sh Qwen/Qwen3.5-27B 8000 -- --tensor-parallel-size 2
#   scripts/run-vllm.sh --dry-run baselabs/mammoth-1 8001
#
# 额外参数放在 -- 之后，原样传给 vllm serve。

set -euo pipefail

IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:v0.26.0}"
DRY_RUN=0

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi

if [[ $# -lt 1 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "用法: scripts/run-vllm.sh [--dry-run] MODEL [宿主端口] [-- 额外的 vllm serve 参数]" >&2
  exit 2
fi

MODEL="$1"
HOST_PORT="${2:-8000}"
if [[ $# -ge 2 ]]; then
  shift 2
else
  shift
fi
if [[ "${1:-}" == "--" ]]; then
  shift
fi

if [[ ! "$HOST_PORT" =~ ^[0-9]+$ ]]; then
  echo "宿主端口必须是整数: ${HOST_PORT}" >&2
  exit 2
fi

NAME="vllm-${HOST_PORT}"
HF_CACHE="${HF_CACHE:-${HOME}/.cache/huggingface}"

cmd=(
  docker run -d
  --name "$NAME"
  --runtime nvidia
  --gpus all
  --ipc=host
  -p "${HOST_PORT}:8000"
  -v "${HF_CACHE}:/root/.cache/huggingface"
)
if [[ -n "${HF_TOKEN:-}" ]]; then
  cmd+=(--env "HF_TOKEN=${HF_TOKEN}")
fi
cmd+=(
  "$IMAGE"
  --model "$MODEL"
  --host 0.0.0.0
  --port 8000
  --api-key EMPTY
  --enable-prefix-caching
  --max-logprobs 26
  "$@"
)

if [[ "$DRY_RUN" -eq 1 ]]; then
  printf '%q ' "${cmd[@]}"
  printf '\n'
  exit 0
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "未找到 docker。" >&2
  exit 1
fi

if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  echo "容器 ${NAME} 已存在。确认不再使用后执行: docker rm -f ${NAME}" >&2
  exit 1
fi

mkdir -p "$HF_CACHE"
docker pull "$IMAGE"
"${cmd[@]}"

echo "容器 ${NAME} 已启动。模型下载和加载期间接口还不可用，可查看日志: docker logs -f ${NAME}"
echo "评测配置里的 base_url 填 http://127.0.0.1:${HOST_PORT}/v1 ，model 填 ${MODEL} ，api_key 填 EMPTY。"
