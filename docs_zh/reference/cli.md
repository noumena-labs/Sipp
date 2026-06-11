# 命令行界面

`apps/cli` 构建 `cogentlm` 命令行应用，支持本地 GGUF 文本生成。适用于运行时冒烟测试、手动模型检查和快速验证本地 Prompt。

## 构建

```bash
cargo xtask build cli --backend cpu
cargo xtask build cli --backend all
```

## 运行

```bash
cargo run -p cogentlm-cli -- <model.gguf> "Explain CogentLM."
```

常用标志：

- `--max-tokens`
- `--ctx-size`
- `--backend auto|cpu|cuda|metal|vulkan`
- `--temperature`
- `--stats off|basic|profile`
- `--chat`

运行 `cargo run -p cogentlm-cli -- --help` 获取完整帮助信息。
