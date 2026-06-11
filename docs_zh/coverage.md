# 测试覆盖率

CogentLM 的测试覆盖率由测试目录（test catalog）驱动，该目录也为 `cargo xtask test list` 提供支持。通用测试命令说明见 [testing.md](testing.md)。

## 命令

```bash
cargo xtask test list
cargo xtask test list --group unit --layer whitebox --cases --format json
cargo xtask test unit group whitebox
cargo xtask test verify --target whitebox
cargo xtask test verify --target node
cargo xtask test verify --changed
```

`test unit` 运行确定性且支持覆盖率的单元测试套件，并生成最新的覆盖率数据。Rust 用 `cargo-llvm-cov` 生成覆盖率，Node 用 `c8`，Python 用 `pytest-cov`。

`test verify` 默认检查所有支持覆盖率的单元测试套件。它不执行测试、不构建绑定、不下载模型，也不运行冒烟测试。用 `--target` 缩小分析范围，只查看已有的覆盖率产物。如果指定了不支持覆盖率的单元测试目标，命令会报错并给出详细信息。

`--changed` 验证修改过的第一方源码文件是否在同属一个目录套件中有对应的修改测试。`test verify` 还会检查目录归属，以及测试代码是否与运行时代码隔离，确保测试代码不会混入运行时代码库。

`test list --format json` 是供 CI 和贡献者使用的稳定目录接口。每个套件条目包含 `id`、`group`、`layer`、`description`、`requirements`、`sourceRoots`、`backendPolicy`、`coverage` 和 `caseDiscovery`。如果工具需要映射到套件运行器的测试文件和用例名称，加上 `--cases` 标志。

## 工具

不同语言组件使用各自的覆盖率工具：

- `cargo-llvm-cov` 处理 Rust/原生代码的执行和报告生成。
- `c8` 在运行 `test unit suite node-package` 时收集 Node 包装器的覆盖率数据。
- `pytest-cov` 在运行 `test unit suite python-package` 时收集 Python 包装器的覆盖率数据。

`test verify` 只读取已有的覆盖率产物，生成汇总摘要。

## 输出文件

报告输出到 `.build/coverage/` 目录：

- `rust/lcov.info` 和 `rust/html/`
- `node/lcov.info`
- `python/lcov.info`、`python/cobertura.xml` 和 `python/html/`
- `baseline.json`
- `coverage-summary.md`

测试运行报告输出到 `.build/test/` 目录：

- `run-report.json` 和 `run-report.md`
- `verify-report.json` 和 `verify-report.md`

覆盖率基准（baseline）只包含 `crates/` 和 `bindings/` 目录下的第一方代码，排除了构建输出、缓存、测试、示例和 `third_party/`。

## 覆盖率策略

当前实现仅统计代码基准覆盖率，不会因未达到阈值而报错。但如果某个启用覆盖率的代码区域生成了空的第一方报告，流程会报错拦截。待代码基准趋于稳定、最大面积的未覆盖第一方代码问题解决后，会正式启用阈值限制。
