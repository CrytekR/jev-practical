# jev-practical

这个仓库用来判断：现在用通用模型加提示词做的决策，能不能换成 Jev 类开源决策模型。对比维度是准确率和推理性能。

评测榜的公信力和模型选择写在 [docs/jev-class-leaderboards.md](docs/jev-class-leaderboards.md)。`examples/mammoth_multilabel.py` 演示一次请求里多道题如何共用 state、并在答案位置读取字母 logits。默认只打印提示词；加上 `--run` 才加载权重。

```bash
python3 examples/mammoth_multilabel.py
PYTHONPATH=src python3 -m jev_compare --config experiments/support.json
python3 -m pytest
```

`experiments/support.json` 只写数据路径、本地 vLLM 的地址和模型名、每个模型要跑的 `generate` 或 `jev`，以及并发样本数。数据路径相对配置文件解析。样本是 JSONL 或 JSON 数组，每条有 `task`、`gold`，以及两种方法各自的输入：`generate.messages` 是当前系统的提示词，`jev.state` 和 `jev.questions` 在调用时套成 qev 字母题。

生成请求发到 `{base_url}/chat/completions`。Jev 先用服务的 `/tokenize` 把整段提示词和字母换成 token id，再按题目顺序请求 `{base_url}/completions`，`max_tokens` 为 1，用 `logprob_token_ids` 取字母的对数概率并在字母上归一化。同一条样本的题目顺序发出，后面的题才能用到这条 state 的前缀缓存。并发数是同时在飞的样本数。

汇总按任务、模型、方法写出硬标签准确率、无效率、错误率，以及该并发下每条样本墙钟时间的 p50 和 p95，目录默认是 `runs/`。vLLM 的概率是因果注意力下的部署结果，不会逐位等于 mammoth-1 官方引擎。

本地服务用官方镜像 `vllm/vllm-openai:v0.26.0`，不在本仓库里编译 vLLM。`scripts/run-vllm.sh` 拉取该镜像并创建容器，打开前缀缓存，把 `--max-logprobs` 设为 26，API 密钥设为 `EMPTY`：

```bash
scripts/run-vllm.sh Qwen/Qwen3.5-27B 8000
scripts/run-vllm.sh baselabs/mammoth-1 8001
```
