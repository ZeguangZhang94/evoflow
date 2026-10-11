## 变更说明

<!-- 这个 PR 做了什么、为什么做 -->

## 对应研究结论

<!-- EvoFlow 是研究证据驱动的项目：核心行为变更请给出对应的论文/来源链接；
纯重构、文档、测试改进可填 N/A -->

## 检查清单

- [ ] `python -m pytest tests/ -q` 全绿
- [ ] `python examples/quickstart.py` 仍可完整运行
- [ ] 未给核心包引入第三方依赖（依赖只能进 extras）
- [ ] 未削弱红线：评估门不可绕过、评分器与 agent 隔离、迭代熔断有效、黄金集不进入训练数据
- [ ] 新增/变更行为有对应测试
- [ ] 文档（README / docs/）已同步更新
- [ ] 提交信息遵循 Conventional Commits

## 关联 Issue

Closes #
