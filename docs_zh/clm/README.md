# clm 命令行工具

`clm` 是 CogentLM 代码库的本地启动器。安装后将包装脚本存入 `.build/bin`，随后将所有命令转发给 `cargo xtask`。

本地开发时，构建原生组件、运行示例、启动网关服务、管理 xtask 工具链、执行测试或构建文档，都应使用 `clm`。已发布的包（如 `cogentlm`、`cogentlm-server`、Python wheel）无需 `clm`。

## 命令格式

`clm` 命令参数与 `cargo xtask` 完全一致：

```bash
clm doctor
clm build node --backend cpu
clm run examples serve browser
clm test list
clm docs build
```

当前 Shell 未激活启动器时，可直接用 `cargo xtask` 运行相同命令：

```bash
cargo xtask doctor
cargo xtask build node --backend cpu
```

## 相关页面

- [安装与设置](setup.md)
- [常用命令](commands.md)
- [故障排除](troubleshooting.md)
