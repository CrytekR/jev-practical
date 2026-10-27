# jev-practical

这个仓库用来判断：现在用通用模型加提示词做的决策，能不能换成 Jev 类开源决策模型。对比维度是准确率和推理性能。

评测榜的公信力和模型选择写在 [docs/jev-class-leaderboards.md](docs/jev-class-leaderboards.md)。`examples/mammoth_multilabel.py` 演示一次请求里多道题如何共用 state、并在答案位置读取字母 logits。默认只打印提示词；加上 `--run` 才加载权重。

```bash
python3 examples/mammoth_multilabel.py
python3 -m pytest
```
