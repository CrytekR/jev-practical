"""对比通用模型文本生成和 Jev 类字母读出。推理都打向本地 vLLM。"""

__all__ = ["main"]


def main() -> None:
    from jev_compare.run import main as run_main

    run_main()
