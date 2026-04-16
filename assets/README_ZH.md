<div align="center">

<img src="./skillclaw_logo.png" alt="SkillClaw" width="150">

# ✨ SkillClaw: Let Skills Evolve Collectively with Agentic Evolver ✨

<h3>让分布式 AI Agent 群体在真实部署中持续进化 —— 只需对话。<br>经验自动提取，技能持续生长，集体智慧即时共享。</h3>

| 🚀 安装快捷 | 💬 只需对话 | 🔌 广泛兼容 | 🧬 技能群体进化 |
|:-:|:-:|:-:|:-:|

[![Hermes](https://img.shields.io/badge/Hermes-supported-ff6b6b?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0id2hpdGUiIGQ9Ik0xMiAyTDIgN2wxMCA1IDEwLTV6TTIgMTdsOSA1VjEyTDIgN3pNMTMgMjJsOS01VjdMMTMgMTJ6Ii8+PC9zdmc+)](https://github.com/NousResearch/hermes-agent)
[![OpenClaw](https://img.shields.io/badge/OpenClaw-supported-6c5ce7?style=flat-square)](https://github.com/openclaw/openclaw)
[![Agents](https://img.shields.io/badge/+QwenPaw%20%7C%20IronClaw%20%7C%20PicoClaw%20%7C%20ZeroClaw%20%7C%20...-8A2BE2?style=flat-square)](https://github.com/AMAP-ML/SkillClaw)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](../LICENSE)
[![Paper](https://img.shields.io/badge/Paper-arXiv-b5212f.svg?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2604.08377)
[![Paper](https://img.shields.io/badge/Paper-PDF-red?style=flat-square&logo=adobeacrobatreader)](https://arxiv.org/pdf/2604.08377)
[![Paper](https://img.shields.io/badge/Paper-Hugging%20Face-yellow?style=flat-square&logo=huggingface)](https://huggingface.co/papers/2604.08377)
[![Docs](https://img.shields.io/badge/Docs-English-blue?style=flat-square)](../README.md)

<br>

<a href="https://github.com/AMAP-ML/SkillClaw"><img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=18&duration=2500&pause=1000&color=58A6FF&center=true&vCenter=true&width=750&lines=AI+Agent+%E6%8A%80%E8%83%BD%E7%BE%A4%E4%BD%93%E8%BF%9B%E5%8C%96%E6%A1%86%E6%9E%B6;Hermes+%C2%B7+OpenClaw+%C2%B7+QwenPaw+%C2%B7+IronClaw+%C2%B7+PicoClaw+%C2%B7+ZeroClaw+%E7%AD%89" alt="Typing SVG" /></a>

<img src="./terminal_cmd.svg" alt="skillclaw setup && skillclaw start --daemon" width="620">

</div>

<br>

<table>
<tr>
<td>🚀 <b>安装快捷</b></td>
<td>macOS/Linux 提供 shell 安装脚本；Windows 提供手动 Python 安装路径。安装后运行 <code>skillclaw setup</code> 和 <code>skillclaw start --daemon</code>。</td>
</tr>
<tr>
<td>💬 <b>只需对话</b></td>
<td>正常和 Agent 对话即可，Skill 的进化在后台无感完成，无需任何额外操作。</td>
</tr>
<tr>
<td>🔌 <b>广泛兼容</b></td>
<td>无缝支持 <a href="https://github.com/NousResearch/hermes-agent">Hermes</a>、<a href="https://github.com/openclaw/openclaw">OpenClaw</a>、QwenPaw、IronClaw、PicoClaw、ZeroClaw、NanoClaw、NemoClaw 及任何 OpenAI 兼容 API。</td>
</tr>
<tr>
<td>🧬 <b>技能群体进化</b></td>
<td>多用户、多 Agent 的实战经验自动提炼为可复用 Skill，通过云端共享，让整个 Agent 集群持续进化。</td>
</tr>
</table>

---

<div align="center">

### 从经验孤岛到群体进化

<img src="./shift_contrast.svg" alt="从经验孤岛到群体进化" width="860">

<br>

### 四个人的经验，一个 Skill 的进化

<img src="./skill_evolution.svg" alt="技能进化流程" width="860">

**N 个用户，一个 Skill，持续进化。** Every conversation compounds. Every user contributes.

</div>

---

<div align="center">

<img src="./skillclaw_main.png" alt="SkillClaw 架构图" width="680">

</div>

---

## 动态

- **2026/04/14** — 微信讨论群已开放！[加入群聊](./image.png)和我们交流。
- **2026/04/14** — 已支持与 [Hermes](https://github.com/NousResearch/hermes-agent) 无缝集成。
- **2026/04/12** — 正在与 [Deer-Flow](https://github.com/bytedance/deer-flow/discussions/2133) 讨论跨框架技能共享。
- **2026/04/11** — SkillClaw 在 [Hugging Face Daily Papers](https://huggingface.co/papers/2604.08377) 上获得**当日第 2 名**！
- **2026/04/10** — SkillClaw 正式开源！代码已发布在 [GitHub](https://github.com/AMAP-ML/SkillClaw)。

---

## 项目概览

SkillClaw 通过从真实会话数据中**演化可复用技能**并在一组 Agent 之间共享，让 LLM Agent 持续变强。

系统由两个部分组成：

1. **Client Proxy** — 本地 API 代理（`/v1/chat/completions`、`/v1/messages`），负责拦截 Agent 请求、记录会话产物，并与共享存储同步技能。

2. **Evolve Server**（`evolve_server`）— 统一的技能演化服务，从共享存储读取会话数据，演化或创建技能，再写回存储。它支持两种 engine：
   - `workflow`：固定三阶段 LLM 流程（Summarize → Aggregate → Execute）
   - `agent`：基于 OpenClaw 的自主 agent 工作区，直接编辑技能文件

两者共享同一套存储层（Alibaba OSS / S3 / 本地文件系统）以及同一种技能格式（`SKILL.md`）。

---

## 部署模型

可以把 SkillClaw 理解成"每个用户一个 Client，每个群组一个 Evolver"：

1. 每个用户都在自己的机器上运行本地 `skillclaw` Client Proxy。
2. 每个共享群组通常只运行一个 `skillclaw-evolve-server`。
3. 用户侧和服务器侧只通过共享存储（`local`、`oss`、`s3`）发生连接。

这个拆分是最适合初学者的理解方式：

- 如果你只是自己用，先装 Client 就够了，后面再决定要不要加 Evolver。
- 如果你是加入一个现有群组，你仍然只需要安装 Client，不需要在自己机器上跑 Evolver。
- Evolver 可以跑在你的本机、远程服务器，或者任何一台能访问共享存储和上游 LLM API 的机器上。


## 用户指南

第一次上手建议先走路径 A，把"用户侧安装和使用"先跑通，不要一开始就把共享部署和服务端混进来。

### 前置要求

- macOS、Linux 或 Windows
- Python >= 3.10
- 一个兼容 OpenAI 的 LLM API endpoint，或 AWS Bedrock
- 只有在你明确选择 `openclaw` CLI 集成或服务端 `agent` engine 时，才需要额外安装 `openclaw`

下面的入门路径已在 macOS 上做过本地烟测。

### 路径 A：先在单机上把用户侧跑通

1. 从本仓库安装 SkillClaw。如果你已经有源码，跳过 `git clone`。

macOS / Linux：

```bash
git clone https://github.com/AMAP-ML/SkillClaw.git && cd SkillClaw
bash scripts/install_skillclaw.sh
source .venv/bin/activate
```

Windows PowerShell（当前仓库没有原生 `.ps1` 安装脚本，所以这里是手动安装方式）：

```powershell
git clone https://github.com/AMAP-ML/SkillClaw.git
Set-Location SkillClaw
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e ".[evolve,sharing,server]"
```

2. 生成本地配置。

```bash
skillclaw setup
```

设置向导会依次提示你选择 provider、模型、本地 skills 目录、PRM 设置、可选的 CLI agent 集成、以及可选的共享存储。

第一次最小化验证时，推荐这样选：

- `CLI agent` 选 `none`，先不要自动改外部 agent 配置
- `skills` 目录保持默认值 `~/.skillclaw/skills`
- 如果你只是想先验证代理能不能正常用，可以先关闭 shared storage
- 如果你后面想在同一台机器上继续跑本地 evolver 闭环，就把 shared storage 打开并选 `local` backend，例如 `~/.skillclaw/local-share`
- 如果你想先把成本压到最低，可以先关闭 PRM

3. 启动 Client Proxy 并检查健康状态。

```bash
skillclaw start --daemon
skillclaw status
PROXY_PORT="$(skillclaw config proxy.port | awk '{print $2}')"
curl "http://127.0.0.1:${PROXY_PORT}/healthz"
```

默认代理端口是 `30000`，但健康检查应以你在配置中实际设定的 `proxy.port` 为准。用 `skillclaw config show` 查看当前的上游模型、代理端口和 sharing 目标。

到这里为止，SkillClaw 已经可以作为"单用户本地代理"使用了。只是想自己用的话，不需要立刻运行 Evolver。

如果你后面想让它自动演化技能，再继续看下面的[服务器指南](#服务器指南)。

### Hermes 接入

如果你本来就在用 Hermes，用户侧的接入路径是：

1. 先安装 Hermes。
2. 运行 `skillclaw setup`，在 `CLI agent to configure` 里选择 `hermes`。
3. `Proxy model name exposed to agents` 保持 `skillclaw-model`，除非你明确知道自己为什么要改它。
4. 启动 SkillClaw。启动时，SkillClaw 会自动改写 `~/.hermes/config.yaml`，把 Hermes 指到本地代理。

最小验证命令：

```bash
skillclaw start --daemon
hermes chat -Q -m skillclaw-model -q "Reply with exactly HERMES_SKILLCLAW_OK and nothing else."
```

### 路径 B：加入一个已有的共享群组

加入群组时，用户侧安装方式和路径 A 完全一样。区别只是把本地 Client 指向群组的共享存储。

最适合初学者的方式是重新执行一次 `skillclaw setup`，开启 shared storage，然后把群组管理员给你的参数填进去。

你也可以直接手动配置。下面是 OSS 示例：

```bash
skillclaw config sharing.enabled true
skillclaw config sharing.backend oss
skillclaw config sharing.endpoint https://oss-cn-hangzhou.aliyuncs.com
skillclaw config sharing.bucket my-skillclaw-bucket
skillclaw config sharing.access_key_id "$OSS_ACCESS_KEY_ID"
skillclaw config sharing.secret_access_key "$OSS_ACCESS_KEY_SECRET"
skillclaw config sharing.group_id my-group
skillclaw config sharing.user_alias alice
skillclaw config sharing.auto_pull_on_start true

skillclaw start --daemon
skillclaw skills pull
```

如果你的群组用的是本地挂载目录或 NAS，而不是 OSS/S3，那么把 `sharing.backend` 改成 `local`，并配置 `sharing.local_root /path/to/shared/root` 即可。

加入群组时要记住：

- 你本机上仍然只运行 Client Proxy
- 你不需要在自己机器上运行 `skillclaw-evolve-server`
- 从单用户扩展到多用户，对用户侧来说主要就是共享存储配置变了

### 可选：把某个客户端打开成后台验证节点

这个模式是可选的，而且默认关闭。它适合那些希望在 `workflow` 产出正式进入共享技能目录前，再加一道群组内复核的部署。

它实际做的是：

- 服务端先把候选技能写成一个 validation job，而不是立刻发布
- 某个选择加入的客户端在本地代理空闲时主动拉取 job
- 客户端会在后台验证这个候选技能，并把结果回写共享存储
- 后续某一轮 evolve cycle 再根据阈值决定正式发布还是拒绝

如果 `validation.enabled` 保持 `false`，你的日常客户端使用路径完全不受影响。

客户端最小配置：

```bash
skillclaw config validation.enabled true
skillclaw config validation.idle_after_seconds 300
skillclaw config validation.poll_interval_seconds 60
skillclaw config validation.max_jobs_per_day 5

skillclaw validation status
skillclaw validation run-once --force
```

之后只要 `skillclaw start --daemon`，后台验证器就会自动跟着启动。`run-once --force` 是最快的自测方式，不需要等空闲计时器。

## 服务器指南

Evolver 是共享后端。它可以服务一个用户，也可以服务一整个群组；可以跑在本机，也可以跑在远程服务器。

### 单用户本地闭环：在同一台机器上运行一个 Evolver

这是最小的完整闭环：一个用户、一台机器、一个本地 shared root、一个 evolve server。

前提是你在 `skillclaw setup` 里已经开启了 `local` shared storage。

```bash
skillclaw-evolve-server --use-skillclaw-config --interval 300 --port 8787
```

跑起来后，你还可以查看共享技能区：

```bash
skillclaw skills list-remote
```

这个路径的特点是：

- 只有一个用户
- 只有一台机器
- 不需要 OSS / S3
- 能完整体验"记录会话 → 演化技能 → 回写复用"的闭环

### 群组共享部署：在本地或远程服务器上运行一个 Evolver

把 Evolver 跑在任何一台能访问共享存储和上游 LLM API 的机器上即可，它不一定要跑在终端用户的电脑上。

macOS / Linux：

```bash
git clone https://github.com/AMAP-ML/SkillClaw.git && cd SkillClaw
bash scripts/install_skillclaw_server.sh
source .venv-server/bin/activate
cp evolve_server/evolve_server.env.example evolve_server/.env
```

Windows PowerShell（当前仓库没有原生 `.ps1` 安装脚本，所以这里是手动安装方式）：

```powershell
git clone https://github.com/AMAP-ML/SkillClaw.git
Set-Location SkillClaw
python -m venv .venv-server
.\.venv-server\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e ".[server]"
Copy-Item .\evolve_server\evolve_server.env.example .\evolve_server\.env
```

然后编辑 `evolve_server/.env`，或者直接通过命令行传参数。默认 `workflow` engine 的 OSS 示例：

```bash
skillclaw-evolve-server --port 8787 --interval 300 \
  --storage-backend oss \
  --oss-endpoint "$EVOLVE_STORAGE_ENDPOINT" \
  --oss-bucket "$EVOLVE_STORAGE_BUCKET" \
  --group-id my-group
```

默认情况下，`workflow` engine 会把演化通过的技能直接写入共享技能目录，也就是 `{group_id}/skills/<name>/SKILL.md`。

如果你想让候选技能先经过一轮"客户端复核"再正式进入 `skills/`，可以把服务端切到 `validated` 发布模式：

```bash
EVOLVE_PUBLISH_MODE=validated \
EVOLVE_VALIDATION_REQUIRED_RESULTS=1 \
EVOLVE_VALIDATION_REQUIRED_APPROVALS=1 \
EVOLVE_VALIDATION_MIN_MEAN_SCORE=0.75 \
EVOLVE_VALIDATION_MAX_REJECTIONS=1 \
skillclaw-evolve-server --port 8787 --interval 300 \
  --storage-backend oss \
  --oss-endpoint "$EVOLVE_STORAGE_ENDPOINT" \
  --oss-bucket "$EVOLVE_STORAGE_BUCKET" \
  --group-id my-group
```

如果你服务端想使用 `agent` engine，那就在服务端机器上安装 `openclaw`，然后运行：

```bash
npm install -g openclaw

skillclaw-evolve-server --engine agent --port 8787 --interval 300 --no-fresh \
  --storage-backend oss \
  --oss-endpoint "$EVOLVE_STORAGE_ENDPOINT" \
  --oss-bucket "$EVOLVE_STORAGE_BUCKET" \
  --group-id my-group
```

只有服务端操作员在使用 `--engine agent` 时才需要 `openclaw`。

## 快速参考

**运行关系** — 每个用户有自己的 `skillclaw` 进程和 `~/.skillclaw/config.yaml`；每个群组对应一个 `group_id`，由一个 Evolver 持续读取会话、产出技能、写回存储。加入群组只需正确配置共享存储，无需部署服务端。

**首次启动检查**

| 命令 | 预期结果 |
|------|----------|
| `skillclaw status` | 显示 `running` |
| `curl http://127.0.0.1:<port>/healthz` | 返回 `{"ok": true}` |
| `skillclaw config show` | 显示正确的上游 URL、模型和 sharing 配置 |

**技能管理**

```bash
skillclaw skills pull          # 下载共享技能
skillclaw skills push          # 上传本地技能
skillclaw skills sync          # 双向同步
skillclaw skills list-remote   # 查看远端技能
```

**配置参考**

- Client 配置：`~/.skillclaw/config.yaml`（由 `skillclaw setup` 生成）
- 服务端模板：[`evolve_server/evolve_server.env.example`](../evolve_server/evolve_server.env.example)（复制为 `.env` 使用）
- 查看 / 修改配置：`skillclaw config show` ｜ `skillclaw config <key> <value>`
- 贡献者入口：`skillclaw/`（客户端）、`evolve_server/`（后端）、`scripts/`（安装脚本）


## 致谢

本仓库基于以下开源项目构建：

- [MetaClaw](https://github.com/aiming-lab/MetaClaw) - Just talk to your agent — it learns and evolves
- [WildClawBench](https://github.com/InternLM/WildClawBench) - Can an AI agent do real work, end-to-end, without hand-holding
- [OpenClaw-RL](https://github.com/Gen-Verse/OpenClaw-RL) - Train a personalized agent simply by talking to it

## 共建

SkillClaw 是一个社区共建项目。我们欢迎各种形式的贡献——Bug 反馈、功能建议、新技能贡献、文档完善等。欢迎提 Issue 或提交 Pull Request！

## 引用

如果 SkillClaw 对你的研究有帮助，请考虑引用我们的论文：

```bibtex
@article{ma2026skillclaw,
  title={SkillClaw: Let Skills Evolve Collectively with Agentic Evolver},
  author={Ma, Ziyu and Yang, Shidong and Ji, Yuxiang and Wang, Xucong and Wang, Yong and Hu, Yiming and Huang, Tongwen and Chu, Xiangxiang},
  journal={arXiv preprint arXiv:2604.08377},
  year={2026}
}
```

## 许可证

详见 [LICENSE](../LICENSE)。
