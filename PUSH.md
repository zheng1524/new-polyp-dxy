# 推送到新的私有 GitHub 仓库

本地仓库已在 `main` 分支完成冻结提交：`9ce1c7c`。

创建一个新的**私有** GitHub 仓库后，在本目录执行：

```bash
git remote add origin <new-repository-url>
git push -u origin main
```

推荐仓库名：`dxy-s5-cpag39-mainline`。

不要上传原始视频、完整帧缓存、模型权重或患者数据；本仓库已刻意排除这些内容。
