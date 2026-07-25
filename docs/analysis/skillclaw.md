# SkillClaw 项目速读（初学者版）

## 它到底是做什么的？

`SkillClaw` 的核心目标是：把你和 Agent 的真实对话数据变成可复用、可共享、可持续进化的技能（`SKILL.md`），让后续对话变得更顺、更快、更准。  

你可以把它理解成两层系统：

1. **本地客户端（client）**：拦截你本地和 Agent 的 API 调用，记录会话并管理技能库，保证你现在就能用。
2. **进化服务（evolve server，可选）**：周期性读取会话记录，生成/优化/验证技能，再写回共享存储，让多人共享时技能持续升级。

## 代码结构一眼看懂

- `skillclaw/`：客户端主模块，包含 CLI、代理服务、配置、技能管理、验证、Dashboard 等能力。
- `evolve_server/`：服务端模块，包含 CLI、存储层、pipeline、engine（workflow / agent）等。
- `scripts/`：安装脚本（客户端和服务端）与示例脚本。
- `assets/`：README/文档使用的图片与资源。
- `tests/`：完整测试覆盖客户端与服务端核心行为。
- 根目录配置文件：`requirements*.txt`、`pyproject.toml`。

## 执行流程（先用起来）

1. 用 `skillclaw setup` 生成本地配置（上游模型、端口、技能目录、共享存储可选项）。
2. 用 `skillclaw start --daemon` 启动本地代理。
3. 与 Agent 正常对话：SkillClaw 后台采集 session / skill 产物。
4. 如果需要自动进化，另外启动 `skillclaw-evolve-server`，让它在共享存储上持续产出新技能版本。
5. 用 `skillclaw skills`、`dashboard` 等命令查看效果。

## 为什么有用（给初学者的直觉）

- 先不用改你原有使用习惯，几乎零感知接入。
- 技能文件是可读文本（`SKILL.md`），你能看到系统在学什么、学成什么。
- 多设备/多 Agent/多人共享时，知识可以沉淀到同一技能体系，减少重复踩坑。

## 先别慌：建议的第一步

- 第一步只装客户端，把 `skillclaw setup` + `skillclaw start --daemon` 跑通。
- 如果只是个人使用，先不开启共享/服务端也能正常工作。
- 想体验群体进化，再按官方文档补上 `evolve_server` 与共享存储配置。
