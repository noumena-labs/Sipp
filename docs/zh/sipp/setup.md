# 安装与设置

在项目根目录下运行安装脚本。它会在需要时构建 `xtask` 二进制文件，将 `sipp` 启动器安装到 `.build/bin`，并为选定的工作流准备工具链和示例文件。

## Unix Shell

```bash
source ./setup.sh
sipp doctor
```

直接运行 `./setup.sh`（不带 `source`）仍会执行安装，但不会修改当前 Shell 的 `PATH`。运行结束后会提示你执行 `source` 来加载环境脚本。

## Windows PowerShell

```powershell
.\setup.ps1
sipp doctor
```

PowerShell 脚本会更新当前会话的 `PATH`，安装成功后加载 `.build\bin\sipp-env.ps1`。

## Windows CMD

```bat
setup.cmd
sipp doctor
```

`setup.cmd` 调用 PowerShell 安装脚本，并为当前 CMD 会话激活 `.build\bin`。

## 配置预设

如果已明确开发方向，使用对应的预设：

```bash
sipp setup --profile browser
sipp setup --profile bindings
sipp setup --profile full --yes
```

| 预设 | 适用场景 |
| --- | --- |
| `browser` | 开发浏览器包、WASM、WebGPU 示例及演示项目。 |
| `bindings` | 开发原生 Node.js 和 Python 绑定。 |
| `full` | 包含浏览器和原生绑定的全工作区。 |

常用安装参数：

- `--yes`：接受推荐操作，不弹出交互提示。
- `--no-downloads`：跳过工具链、依赖和示例模型下载。
- `--no-splash`：跳过启动动画。
- `--plain`：禁用终端渲染。

## 生成的文件

安装程序只将文件写入工作区本地：

- `.build/xtask/debug/xtask` 或 `.build\xtask\debug\xtask.exe`
- `.build/bin/sipp`、`.build/bin/sipp.cmd`、`.build/bin/sipp.ps1`
- `.build/bin/sipp-env.sh`、`.build/bin/sipp-env.ps1`
- `.build/toolchain` 下由 xtask 管理的工具链和缓存文件
