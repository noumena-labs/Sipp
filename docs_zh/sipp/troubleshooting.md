# 故障排除

## 未找到 sipp 命令

在项目根目录运行安装脚本，并确保在当前 Shell 中激活了环境：

```bash
source ./setup.sh
```

```powershell
.\setup.ps1
```

```bat
setup.cmd
```

`sipp` 无法激活时，可直接用 `cargo xtask` 执行相同命令：

```bash
cargo xtask doctor
cargo xtask test list
```

## 安装程序重复构建 xtask

如果 xtask 源文件、工作区清单或 Cargo 配置的修改时间晚于 `.build/xtask/sipp.stamp`，安装脚本会重新构建 `.build/xtask/debug/xtask`。拉取含构建脚本更新的代码后，这是正常行为。

## PowerShell 阻止脚本执行

调整当前用户的执行策略，或以绕过策略的方式启动脚本：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1
```

## PATH 仅在当前终端生效

启动器安装在 `.build/bin` 目录，安装程序只为当前会话激活了该目录。打开新终端后，重新运行安装程序或执行生成的环境脚本：

```bash
source .build/bin/sipp-env.sh
```

```powershell
. .build\bin\sipp-env.ps1
```

## 缺失工具链或后端

运行健康检查：

```bash
sipp doctor
sipp toolchain status
```

然后按需安装缺失的组件：

```bash
sipp toolchain install uv
sipp toolchain install all
```

注意：CUDA 不由 xtask 自动安装。请从 NVIDIA 官方渠道安装 CUDA，然后重新运行 `sipp doctor --target node --backend cuda` 检查。
