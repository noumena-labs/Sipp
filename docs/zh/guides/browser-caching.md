# 浏览器缓存

运行时环境支持时，浏览器本地推理会将模型数据缓存到浏览器存储中，避免后续加载时重复大文件网络下载。Sipp 的浏览器示例和演示项目均使用此机制加载 GGUF 模型。

## 职责划分

浏览器包负责实现运行时集成和缓存机制。应用仍需负责以下事项：

- 提供选择模型 URL 或文件的 UI。
- 处理进度显示和任务取消。
- 提供清除存储的控制机制，方便用户释放空间。
- 浏览器存储不可用时提供合理的回退逻辑。

## 最佳实践

- 大模型优先使用支持范围请求（Range Requests）的 URL。
- 提供足够小的默认模型，让用户快速完成首次体验。
- 浏览器存储是"尽力而为"且受用户直接控制的空间，应用不应假定缓存永远存在。
- 页面、Worker 或组件不再需要本地计算资源时，务必关闭 `SippClient` 实例以释放内存。

## 下载中断与续传

远程模型下载遇到网络停滞或可重试请求失败时，会保留已经写入 OPFS 的字节。后续尝试会使用 `Range` 请求；服务器提供 ETag 或最后修改时间时还会发送 `If-Range`。如果服务器拒绝或忽略范围请求，Sipp 会删除部分文件并从头重新下载。部分下载会在页面刷新后继续保留；失效或七天内未使用的部分下载会被清理。

添加远程模型时可以设置每个数据块的停滞超时，默认值为 30 秒：

```ts
const model = await client.models.add(['/models/model.gguf'], {
  stallTimeoutMs: 30_000,
});
```

应用可以订阅范围回退事件，无需拦截控制台输出：

```ts
const unsubscribe = client.subscribeEvents((event) => {
  if (event.type === 'fallback-warning' && event.kind === 'transfer') {
    reportDownloadFallback(event.detail);
  }
});

// 所属视图或 Worker 释放时停止订阅。
unsubscribe();
```

最小化实现流程参考浏览器示例代码；运行时诊断使用操场工具（playground）。
