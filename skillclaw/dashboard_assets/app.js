const LOCALE_STORAGE_KEY = "skillclaw.dashboard.locale"

function initialLocale() {
  try {
    const stored = window.localStorage.getItem(LOCALE_STORAGE_KEY)
    if (stored === "en" || stored === "zh") {
      return stored
    }
  } catch {}
  return "zh"
}

const state = {
  activeView: "overview",
  loading: false,
  locale: initialLocale(),
  overview: null,
  evolve: null,
  skills: [],
  sessions: [],
  validationJobs: [],
  skillDetails: Object.create(null),
  sessionDetails: Object.create(null),
  compareSelections: Object.create(null),
  selectedLocalSkillId: "",
  selectedCandidateJobId: "",
  selectedFinalSkillId: "",
  selectedSessionId: "",
}

const dom = {
  body: document.body,
  messageStrip: document.querySelector("#message-strip"),
  brandTitle: document.querySelector("#brand-title"),
  brandCopy: document.querySelector("#brand-copy"),
  navOverviewTitle: document.querySelector("#nav-overview-title"),
  navLocalTitle: document.querySelector("#nav-local-title"),
  navCandidateTitle: document.querySelector("#nav-candidate-title"),
  navFinalTitle: document.querySelector("#nav-final-title"),
  navSessionsTitle: document.querySelector("#nav-sessions-title"),
  navButtons: Array.from(document.querySelectorAll("[data-view]")),
  viewPanels: Array.from(document.querySelectorAll("[data-view-panel]")),
  navOverviewMeta: document.querySelector("#nav-overview-meta"),
  navLocalMeta: document.querySelector("#nav-local-meta"),
  navCandidateMeta: document.querySelector("#nav-candidate-meta"),
  navFinalMeta: document.querySelector("#nav-final-meta"),
  navSessionsMeta: document.querySelector("#nav-sessions-meta"),
  sidebarStatusTitle: document.querySelector("#sidebar-status-title"),
  sidebarStatus: document.querySelector("#sidebar-status"),
  heroKicker: document.querySelector("#hero-kicker"),
  heroTitle: document.querySelector("#hero-title"),
  localeButton: document.querySelector("#btn-locale"),
  refreshButton: document.querySelector("#btn-refresh"),
  syncButton: document.querySelector("#btn-sync"),
  overviewKicker: document.querySelector("#overview-kicker"),
  overviewTitle: document.querySelector("#overview-title"),
  overviewCopy: document.querySelector("#overview-copy"),
  watchlistKicker: document.querySelector("#watchlist-kicker"),
  watchlistTitle: document.querySelector("#watchlist-title"),
  scopeKicker: document.querySelector("#scope-kicker"),
  scopeTitle: document.querySelector("#scope-title"),
  eventsKicker: document.querySelector("#events-kicker"),
  eventsTitle: document.querySelector("#events-title"),
  eventsCopy: document.querySelector("#events-copy"),
  overviewMetrics: document.querySelector("#overview-metrics"),
  overviewWatchlist: document.querySelector("#overview-watchlist"),
  overviewContext: document.querySelector("#overview-context"),
  overviewEvents: document.querySelector("#overview-events"),
  localKicker: document.querySelector("#local-kicker"),
  localTitle: document.querySelector("#local-title"),
  localCopy: document.querySelector("#local-copy"),
  localSearch: document.querySelector("#local-search"),
  localList: document.querySelector("#local-list"),
  localDetail: document.querySelector("#local-detail"),
  candidateKicker: document.querySelector("#candidate-kicker"),
  candidateTitle: document.querySelector("#candidate-title"),
  candidateCopy: document.querySelector("#candidate-copy"),
  candidateSearch: document.querySelector("#candidate-search"),
  candidateStatus: document.querySelector("#candidate-status"),
  candidateStatusAll: document.querySelector("#candidate-status-all"),
  candidateStatusPending: document.querySelector("#candidate-status-pending"),
  candidateStatusReview: document.querySelector("#candidate-status-review"),
  candidateStatusPublished: document.querySelector("#candidate-status-published"),
  candidateStatusRejected: document.querySelector("#candidate-status-rejected"),
  candidateList: document.querySelector("#candidate-list"),
  candidateDetail: document.querySelector("#candidate-detail"),
  finalKicker: document.querySelector("#final-kicker"),
  finalTitle: document.querySelector("#final-title"),
  finalCopy: document.querySelector("#final-copy"),
  finalSearch: document.querySelector("#final-search"),
  finalList: document.querySelector("#final-list"),
  finalDetail: document.querySelector("#final-detail"),
  sessionsKicker: document.querySelector("#sessions-kicker"),
  sessionsTitle: document.querySelector("#sessions-title"),
  sessionsCopy: document.querySelector("#sessions-copy"),
  sessionSearch: document.querySelector("#session-search"),
  sessionSource: document.querySelector("#session-source"),
  sessionSourceLocal: document.querySelector("#session-source-local"),
  sessionSourceAll: document.querySelector("#session-source-all"),
  sessionSourceShared: document.querySelector("#session-source-shared"),
  sessionList: document.querySelector("#session-list"),
  sessionDetail: document.querySelector("#session-detail"),
  opButtons: Array.from(document.querySelectorAll("[data-op]")),
}

function localeTag() {
  return state.locale === "en" ? "en-US" : "zh-CN"
}

function l(zh, en, vars = {}) {
  const template = state.locale === "en" ? en : zh
  return String(template).replace(/\{(\w+)\}/g, (_, key) => String(vars[key] ?? ""))
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;")
}

function clip(value, limit = 160) {
  const text = String(value ?? "").trim().replace(/\s+/g, " ")
  if (text.length <= limit) {
    return text
  }
  return `${text.slice(0, limit).trimEnd()}...`
}

function number(value) {
  return Number(value || 0).toLocaleString(localeTag())
}

function parseTime(value) {
  const timestamp = Date.parse(String(value || ""))
  return Number.isFinite(timestamp) ? timestamp : 0
}

function formatStamp(value) {
  const timestamp = parseTime(value)
  if (!timestamp) {
    return l("无时间", "No time")
  }
  return new Intl.DateTimeFormat(localeTag(), {
    year: "numeric",
    month: "numeric",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(timestamp))
}

function formatScore(value) {
  if (value == null || value === "") {
    return "-"
  }
  const numeric = Number(value)
  if (!Number.isFinite(numeric)) {
    return "-"
  }
  return numeric.toFixed(2)
}

function shortHash(value, size = 10) {
  const text = String(value || "").trim()
  if (!text) {
    return "-"
  }
  return text.length <= size ? text : text.slice(0, size)
}

function getJson(url, options = {}) {
  return fetch(url, {
    headers: {
      "Content-Type": "application/json",
    },
    ...options,
  }).then(async (response) => {
    const payload = await response.json().catch(() => ({}))
    if (!response.ok) {
      throw new Error(String(payload?.detail || payload?.error || response.statusText || l("请求失败", "Request failed")))
    }
    return payload
  })
}

function showMessage(kind, text) {
  dom.messageStrip.className = `message-strip ${kind || "info"}`
  dom.messageStrip.textContent = text
}

function clearMessage() {
  dom.messageStrip.className = "message-strip hidden"
  dom.messageStrip.textContent = ""
}

function setLoading(value) {
  state.loading = Boolean(value)
  dom.body.dataset.loading = value ? "true" : "false"
  for (const button of dom.opButtons) {
    button.disabled = Boolean(value)
  }
}

function tag(label) {
  return `<span class="tag">${escapeHtml(label)}</span>`
}

function badge(label, tone = "neutral") {
  return `<span class="status-badge ${escapeHtml(tone)}">${escapeHtml(label)}</span>`
}

function actionLabel(action) {
  const normalized = String(action || "").trim().toLowerCase()
  if (!normalized) {
    return l("未标注", "Unlabeled")
  }
  if (normalized === "create" || normalized === "create_skill") {
    return l("创建", "Create")
  }
  if (normalized === "improve") {
    return l("改进", "Improve")
  }
  if (normalized === "merge") {
    return l("合并", "Merge")
  }
  if (normalized === "published_after_validation") {
    return l("验证后发布", "Publish After Validation")
  }
  if (normalized === "snapshot") {
    return l("快照", "Snapshot")
  }
  return String(action)
}

function outcomeLabel(outcome) {
  const normalized = String(outcome || "").trim().toLowerCase()
  if (!normalized) {
    return l("未标注", "Unlabeled")
  }
  if (normalized === "success") {
    return l("成功", "Success")
  }
  if (normalized === "review") {
    return l("待复核", "Needs Review")
  }
  if (normalized === "rollback") {
    return l("回滚", "Rolled Back")
  }
  if (normalized === "failure") {
    return l("失败", "Failure")
  }
  return String(outcome)
}

function sourceLabel(source) {
  const normalized = String(source || "").trim().toLowerCase()
  if (normalized === "local") {
    return l("本地", "Local")
  }
  if (normalized === "shared") {
    return l("共享库", "Shared Pool")
  }
  if (normalized === "both") {
    return l("本地和共享库", "Local + Shared Pool")
  }
  if (normalized === "observed") {
    return l("仅观测记录", "Observed Only")
  }
  return String(source || "-")
}

function categoryLabel(value) {
  const normalized = String(value || "").trim().toLowerCase()
  if (!normalized) {
    return l("未分类", "Uncategorized")
  }
  const labels = {
    general: l("通用", "General"),
    candidate: l("候选", "Candidate"),
    coding: l("编码", "Coding"),
    communication: l("沟通", "Communication"),
    security: l("安全", "Security"),
    automation: l("自动化", "Automation"),
    agentic: l("代理", "Agentic"),
    common_mistakes: l("常见错误", "Common Mistakes"),
    research: l("研究", "Research"),
    devops: l("运维", "DevOps"),
    productivity: l("效率", "Productivity"),
    data_analysis: l("数据分析", "Data Analysis"),
  }
  return labels[normalized] || String(value)
}

function candidateStatusKey(value) {
  const normalized = String(value || "").trim().toLowerCase()
  if (normalized === "pending_validation" || normalized === "pending") {
    return "pending"
  }
  if (normalized === "review") {
    return "review"
  }
  if (normalized === "published") {
    return "published"
  }
  if (normalized === "rejected") {
    return "rejected"
  }
  return normalized || "pending"
}

function candidateStatusLabel(value) {
  const status = candidateStatusKey(value)
  if (status === "pending") {
    return l("待验证", "Awaiting Validation")
  }
  if (status === "review") {
    return l("已有反馈", "Feedback Received")
  }
  if (status === "published") {
    return l("已入最终池", "Published")
  }
  if (status === "rejected") {
    return l("已拒绝", "Rejected")
  }
  return String(value || l("未知状态", "Unknown Status"))
}

function toneForStatus(value) {
  const status = candidateStatusKey(value)
  if (status === "published") {
    return "published"
  }
  if (status === "rejected") {
    return "rejected"
  }
  if (status === "pending" || status === "review") {
    return "pending"
  }
  return "neutral"
}

function eventTypeLabel(value) {
  const normalized = String(value || "").trim().toLowerCase()
  if (normalized === "candidate") {
    return l("候选生成", "Candidate Created")
  }
  if (normalized === "validation") {
    return l("验证反馈", "Validation Feedback")
  }
  if (normalized === "publish") {
    return l("正式发布", "Published")
  }
  if (normalized === "reject") {
    return l("拒绝决策", "Rejected")
  }
  return String(value || l("事件", "Event"))
}

function stageStateLabel(value) {
  const normalized = String(value || "").trim().toLowerCase()
  if (normalized === "done") {
    return l("已完成", "Done")
  }
  if (normalized === "current") {
    return l("进行中", "In Progress")
  }
  if (normalized === "pending") {
    return l("待开始", "Pending")
  }
  if (normalized === "blocked") {
    return l("已终止", "Blocked")
  }
  return String(value || "-")
}

function normalizeSkillNames(items) {
  if (!Array.isArray(items)) {
    return []
  }
  const result = []
  for (const item of items) {
    if (typeof item === "string" && item.trim()) {
      result.push(item.trim())
      continue
    }
    if (item && typeof item === "object") {
      const raw = item.skill_name || item.name || item.skill
      if (String(raw || "").trim()) {
        result.push(String(raw).trim())
      }
    }
  }
  return [...new Set(result)]
}

function sortSkills(items) {
  return [...items].sort((left, right) => (
    Number(right.session_count || 0) - Number(left.session_count || 0)
    || Number(right.observed_injection_count || 0) - Number(left.observed_injection_count || 0)
    || Number(right.local_inject_count || 0) - Number(left.local_inject_count || 0)
    || parseTime(right.updated_at || right.uploaded_at) - parseTime(left.updated_at || left.uploaded_at)
    || String(left.name || "").localeCompare(String(right.name || ""), "zh-CN")
  ))
}

function sortSessions(items) {
  return [...items].sort((left, right) => (
    parseTime(right.timestamp) - parseTime(left.timestamp)
    || String(left.session_id || "").localeCompare(String(right.session_id || ""), "zh-CN")
  ))
}

function sortJobs(items) {
  const priority = {
    pending: 0,
    review: 1,
    published: 2,
    rejected: 3,
  }
  return [...items].sort((left, right) => (
    (priority[candidateStatusKey(left.status)] ?? 99) - (priority[candidateStatusKey(right.status)] ?? 99)
    || parseTime(right.created_at) - parseTime(left.created_at)
    || String(left.skill_name || "").localeCompare(String(right.skill_name || ""), "zh-CN")
  ))
}

function localSkillItems() {
  return sortSkills(
    state.skills.filter((item) => item?.has_local || String(item?.source || "") === "local" || String(item?.source || "") === "both")
  )
}

function finalSkillItems() {
  return sortSkills(
    state.skills.filter((item) => item?.has_remote || String(item?.source || "") === "shared" || String(item?.source || "") === "both")
  )
}

function allSessionItems() {
  return sortSessions(state.sessions)
}

function mySessionItems() {
  return sortSessions(state.sessions.filter((item) => String(item?.source || "").trim().toLowerCase() === "local"))
}

function candidateItems() {
  return sortJobs(state.validationJobs)
}

function findSkillById(skillId) {
  return state.skills.find((item) => String(item.skill_id || "") === String(skillId || "")) || null
}

function findSkillByName(name) {
  const normalized = String(name || "").trim().toLowerCase()
  if (!normalized) {
    return null
  }
  return state.skills.find((item) => String(item.name || "").trim().toLowerCase() === normalized) || null
}

function findJobById(jobId) {
  return state.validationJobs.find((item) => String(item.job_id || "") === String(jobId || "")) || null
}

function findSessionById(sessionId) {
  return state.sessions.find((item) => String(item.session_id || "") === String(sessionId || "")) || null
}

function selectedLocalSkillDetail() {
  return state.skillDetails[state.selectedLocalSkillId] || null
}

function selectedFinalSkillDetail() {
  return state.skillDetails[state.selectedFinalSkillId] || null
}

function selectedCandidateJob() {
  return findJobById(state.selectedCandidateJobId)
}

function selectedSessionDetail() {
  return state.sessionDetails[state.selectedSessionId] || null
}

function sharingEnabled() {
  return Boolean(state.overview?.meta?.sharing_enabled)
}

function sharingTarget() {
  const meta = state.overview?.meta || {}
  if (!meta.sharing_enabled) {
    return l("未启用共享", "Sharing Disabled")
  }
  const backend = String(meta.sharing_backend || "").trim().toLowerCase()
  const groupId = String(meta.sharing_group_id || "default").trim()
  if (backend === "local") {
    return `${String(meta.sharing_local_root || "").trim() || "-"} / ${groupId}`
  }
  return `${backend || "shared"} / ${groupId}`
}

function warnings() {
  return Array.isArray(state.overview?.meta?.warnings) ? state.overview.meta.warnings : []
}

function evolveHealthLabel() {
  if (!state.evolve?.configured) {
    return l("未配置", "Not Configured")
  }
  return state.evolve?.healthy ? l("正常", "Healthy") : l("异常", "Unhealthy")
}

function compareLocalAndRemote(skill) {
  if (!skill?.has_local) {
    return { key: "shared_only", label: l("仅在共享库", "Shared Only"), tone: "neutral" }
  }
  if (!skill?.has_remote) {
    return {
      key: sharingEnabled() ? "not_published" : "sharing_disabled",
      label: sharingEnabled() ? l("尚未发布到共享库", "Not Published to Shared Pool") : l("共享未启用", "Sharing Disabled"),
      tone: "neutral",
    }
  }
  if (skill.local_sha && skill.remote_sha && skill.local_sha === skill.remote_sha) {
    return { key: "synced", label: l("已与共享正式版同步", "Synced with Shared Official Version"), tone: "published" }
  }
  return {
    key: "drift",
    label: Number(skill.current_version || 0) > 0
      ? l("与共享版 v{version} 不一致", "Different from shared v{version}", { version: number(skill.current_version) })
      : l("与共享版不一致", "Different from shared version"),
    tone: "pending",
  }
}

function candidateRecordCount(skillName) {
  return jobsForSkill(skillName).length
}

function sharedVersionLabel(skill) {
  return skill.has_remote ? `v${number(skill.current_version || 0)}` : l("未发布", "Not Published")
}

function versionCount(skill) {
  return Array.isArray(skill?.versions) ? skill.versions.length : 0
}

function visibleVersionCount(skill) {
  return Math.max(
    versionCount(skill),
    Number(skill?.current_version || 0),
    skill?.has_remote ? 1 : 0
  )
}

function jobPayload(job) {
  return job?.details?.job || {}
}

function jobResults(job) {
  return Array.isArray(job?.details?.results) ? job.details.results : []
}

function jobDecision(job) {
  return job?.details?.decision || {}
}

function jobSessionIds(job) {
  const ids = jobPayload(job).session_ids
  return Array.isArray(ids) ? ids.filter(Boolean) : []
}

function jobsForSkill(skillName) {
  const normalized = String(skillName || "").trim().toLowerCase()
  return sortJobs(
    state.validationJobs.filter((job) => String(job.skill_name || "").trim().toLowerCase() === normalized)
  )
}

function jobsForSession(sessionId) {
  return sortJobs(
    state.validationJobs.filter((job) => jobSessionIds(job).includes(sessionId))
  )
}

function filteredLocalSkills() {
  const search = String(dom.localSearch.value || "").trim().toLowerCase()
  return localSkillItems().filter((skill) => {
    if (!search) {
      return true
    }
    return [
      skill.name,
      skill.description,
      skill.category,
    ].some((field) => String(field || "").toLowerCase().includes(search))
  })
}

function filteredCandidateJobs() {
  const search = String(dom.candidateSearch.value || "").trim().toLowerCase()
  const status = String(dom.candidateStatus.value || "").trim().toLowerCase()
  return candidateItems().filter((job) => {
    if (status && candidateStatusKey(job.status) !== status) {
      return false
    }
    if (!search) {
      return true
    }
    const details = jobPayload(job)
    return [
      job.job_id,
      job.skill_name,
      job.proposed_action,
      details.rationale,
      details.source,
    ].some((field) => String(field || "").toLowerCase().includes(search))
  })
}

function filteredFinalSkills() {
  const search = String(dom.finalSearch.value || "").trim().toLowerCase()
  return finalSkillItems().filter((skill) => {
    if (!search) {
      return true
    }
    return [
      skill.name,
      skill.description,
      skill.category,
      skill.uploaded_by,
    ].some((field) => String(field || "").toLowerCase().includes(search))
  })
}

function filteredSessions() {
  const search = String(dom.sessionSearch.value || "").trim().toLowerCase()
  const source = String(dom.sessionSource.value || "local").trim().toLowerCase()
  let items = allSessionItems()
  if (source === "local") {
    items = items.filter((item) => String(item.source || "").trim().toLowerCase() === "local")
  } else if (source === "shared") {
    items = items.filter((item) => String(item.source || "").trim().toLowerCase() === "shared")
  }
  return items.filter((item) => {
    if (!search) {
      return true
    }
    return [
      item.session_id,
      item.prompt_preview,
      item.response_preview,
      item.user_alias,
      ...(Array.isArray(item.skill_names) ? item.skill_names : []),
    ].some((field) => String(field || "").toLowerCase().includes(search))
  })
}

function candidateCounts() {
  const jobs = candidateItems()
  return {
    total: jobs.length,
    pending: jobs.filter((item) => candidateStatusKey(item.status) === "pending").length,
    review: jobs.filter((item) => candidateStatusKey(item.status) === "review").length,
    published: jobs.filter((item) => candidateStatusKey(item.status) === "published").length,
    rejected: jobs.filter((item) => candidateStatusKey(item.status) === "rejected").length,
  }
}

function localSyncCounts() {
  const skills = localSkillItems()
  let synced = 0
  let drift = 0
  let localOnly = 0
  for (const skill of skills) {
    if (!skill?.has_remote) {
      localOnly += 1
      continue
    }
    const status = compareLocalAndRemote(skill)
    if (status.key === "synced") {
      synced += 1
    } else {
      drift += 1
    }
  }
  return { synced, drift, localOnly }
}

function collectPipelineEvents(limit = 12) {
  const events = []
  for (const job of candidateItems()) {
    events.push({
      type: "candidate",
      tone: "neutral",
      timestamp: job.created_at,
      title: l("{name} 进入候选池", "{name} entered the candidate pool", {
        name: job.skill_name || l("未知技能", "Unknown Skill"),
      }),
      copy: `${actionLabel(job.proposed_action)} · ${job.job_id}`,
      jobId: job.job_id,
    })
    for (const result of jobResults(job)) {
      const accepted = result?.accepted === true
      events.push({
        type: "validation",
        tone: accepted ? "published" : "rejected",
        timestamp: result.created_at,
        title: l("{name} 提交验证结果", "{name} submitted validation feedback", {
          name: result.user_alias || l("验证客户端", "Validation Client"),
        }),
        copy: l("{skill} · {decision} · 分数 {score}", "{skill} · {decision} · score {score}", {
          skill: job.skill_name || l("未知技能", "Unknown Skill"),
          decision: accepted ? l("通过", "Approved") : l("拒绝", "Rejected"),
          score: formatScore(result.score),
        }),
        jobId: job.job_id,
      })
    }
    const decision = jobDecision(job)
    if (decision?.status && decision?.decided_at) {
      const published = candidateStatusKey(decision.status) === "published"
      events.push({
        type: published ? "publish" : "reject",
        tone: published ? "published" : "rejected",
        timestamp: decision.decided_at,
        title: published
          ? l("{name} 进入最终池", "{name} was published", { name: job.skill_name || l("未知技能", "Unknown Skill") })
          : l("{name} 被拒绝", "{name} was rejected", { name: job.skill_name || l("未知技能", "Unknown Skill") }),
        copy: String(decision.reason || decision.published_action || job.proposed_action || ""),
        jobId: job.job_id,
      })
    }
  }
  return events
    .sort((left, right) => parseTime(right.timestamp) - parseTime(left.timestamp))
    .slice(0, limit)
}

function recentEventCount(hours = 24) {
  const threshold = Date.now() - hours * 60 * 60 * 1000
  return collectPipelineEvents(200).filter((event) => parseTime(event.timestamp) >= threshold).length
}

function validationDispatchSummary(job) {
  const core = jobPayload(job)
  const results = jobResults(job)
  const minResults = Math.max(0, Number(core.min_results || 0))
  const minApprovals = Math.max(0, Number(core.min_approvals || 0))
  const accepted = Math.max(0, Number(job.accepted_count || 0))
  const pendingResults = Math.max(0, minResults - results.length)
  const pendingApprovals = Math.max(0, minApprovals - accepted)
  const lastResult = [...results].sort((left, right) => parseTime(right.created_at) - parseTime(left.created_at))[0] || null
  return {
    dispatchAt: String(job.created_at || ""),
    dispatchMode: "open-pool",
    dispatchLabel: l("开放给空闲验证客户端领取", "Open to available validation clients"),
    pendingResults,
    pendingApprovals,
    responseCount: results.length,
    lastResultAt: String(lastResult?.created_at || ""),
  }
}

function candidateSourceLabel(source) {
  const normalized = String(source || "").trim().toLowerCase()
  if (!normalized) {
    return "-"
  }
  if (normalized === "no_skill") {
    return l("来自会话中新沉淀出的技能", "Newly distilled from sessions")
  }
  if (normalized === "current_skill") {
    return l("来自已有技能的改进", "Improvement of an existing skill")
  }
  if (normalized === "shared_skill") {
    return l("来自共享技能的改进", "Improvement of a shared skill")
  }
  return String(source)
}

function validatorModeLabel(mode) {
  const normalized = String(mode || "").trim().toLowerCase()
  if (!normalized || normalized === "unknown") {
    return l("自动验证", "Automatic Validation")
  }
  if (normalized === "replay") {
    return l("回放验证", "Replay Validation")
  }
  return String(mode)
}

function skillDocumentPreview(skill) {
  if (!skill || typeof skill !== "object") {
    return ""
  }
  if (String(skill.skill_md || "").trim()) {
    return String(skill.skill_md).trim()
  }
  if (String(skill.content || "").trim()) {
    return String(skill.content).trim()
  }
  const parts = []
  if (skill.name) {
    parts.push(`name: ${skill.name}`)
  }
  if (skill.description) {
    parts.push(`description: ${skill.description}`)
  }
  if (skill.category) {
    parts.push(`category: ${skill.category}`)
  }
  return parts.join("\n")
}

function buildVersionEntries(skill, { includeLocal = true } = {}) {
  if (!skill || typeof skill !== "object") {
    return []
  }

  const entries = []
  const seen = new Set()

  const pushEntry = (entry) => {
    if (!entry?.key || seen.has(entry.key)) {
      return
    }
    seen.add(entry.key)
    entries.push(entry)
  }

  const localDocument = String(skill.skill_md || skill.content || "").trim()
  if (includeLocal && skill.has_local && localDocument) {
    pushEntry({
      key: "local-current",
      label: l("本地当前版", "Current Local Version"),
      source: "local",
      version: null,
      action: "local",
      timestamp: String(skill.local_updated_at || skill.updated_at || "").trim(),
      contentSha: String(skill.local_sha || skill.current_sha || "").trim(),
      document: localDocument,
      current: true,
    })
  }

  const versions = Array.isArray(skill.versions) ? skill.versions : []
  for (const item of versions) {
    if (!item || typeof item !== "object") {
      continue
    }
    const version = Number(item.version || 0) || 0
    const contentSha = String(item.content_sha || "").trim()
    pushEntry({
      key: version > 0 ? `shared-version:${version}` : `shared-version:${shortHash(contentSha || item.timestamp || "", 8)}`,
      label: version > 0 ? l("共享库 v{version}", "Shared v{version}", { version }) : l("共享库历史", "Shared History"),
      source: "shared",
      version: version || null,
      action: String(item.action || "").trim(),
      timestamp: String(item.timestamp || "").trim(),
      contentSha,
      document: String(item.skill_md || item.content || "").trim(),
      current: version > 0 && Number(skill.current_version || 0) === version,
    })
  }

  const remoteDocument = String(skill.remote_skill_md || skill.remote_content || "").trim()
  if (skill.has_remote && remoteDocument) {
    const version = Number(skill.current_version || 0) || 0
    pushEntry({
      key: version > 0 ? `shared-version:${version}` : "shared-current",
      label: version > 0 ? l("共享库 v{version}", "Shared v{version}", { version }) : l("共享库当前版", "Current Shared Version"),
      source: "shared",
      version: version || null,
      action: "published",
      timestamp: String(skill.remote_updated_at || skill.uploaded_at || skill.updated_at || "").trim(),
      contentSha: String(skill.remote_sha || skill.current_sha || "").trim(),
      document: remoteDocument,
      current: true,
    })
  }

  const localEntries = entries.filter((item) => item.source === "local")
  const remoteEntries = entries
    .filter((item) => item.source === "shared")
    .sort((left, right) => (
      Number(right.version || 0) - Number(left.version || 0)
      || parseTime(right.timestamp) - parseTime(left.timestamp)
      || String(left.key || "").localeCompare(String(right.key || ""))
    ))

  return [...localEntries, ...remoteEntries]
}

function compareState(skill, scope, { includeLocal = true } = {}) {
  const entries = buildVersionEntries(skill, { includeLocal })
  const stateKey = `${scope}:${skill.skill_id}`
  const current = state.compareSelections[stateKey] || {}
  const defaultPrimary = scope === "local"
    ? (entries.find((item) => item.key === "local-current") || entries.find((item) => item.current) || entries[0] || null)
    : (entries.find((item) => item.current && item.source === "shared") || entries.find((item) => item.source === "shared") || entries[0] || null)
  const defaultCompare = entries.find((item) => item.key !== defaultPrimary?.key) || defaultPrimary || null
  const primary = entries.find((item) => item.key === current.primary) || defaultPrimary
  const compare = entries.find((item) => item.key === current.compare) || defaultCompare
  return { entries, primary, compare }
}

function setCompareSelection(scope, skillId, field, value) {
  const key = `${scope}:${skillId}`
  state.compareSelections = {
    ...state.compareSelections,
    [key]: {
      ...(state.compareSelections[key] || {}),
      [field]: value,
    },
  }
}

function ensureItemSelection(items, currentId, field) {
  if (!items.length) {
    return ""
  }
  if (items.some((item) => String(item?.[field] || "") === String(currentId || ""))) {
    return currentId
  }
  return String(items[0]?.[field] || "")
}

function ensureSelections() {
  state.selectedLocalSkillId = ensureItemSelection(filteredLocalSkills(), state.selectedLocalSkillId, "skill_id")
  state.selectedCandidateJobId = ensureItemSelection(filteredCandidateJobs(), state.selectedCandidateJobId, "job_id")
  state.selectedFinalSkillId = ensureItemSelection(filteredFinalSkills(), state.selectedFinalSkillId, "skill_id")
  state.selectedSessionId = ensureItemSelection(filteredSessions(), state.selectedSessionId, "session_id")
}

async function loadSkillDetail(skillId) {
  if (!skillId) {
    return null
  }
  if (state.skillDetails[skillId]) {
    return state.skillDetails[skillId]
  }
  const payload = await getJson(`/api/v1/skills/${encodeURIComponent(skillId)}`)
  state.skillDetails[skillId] = payload
  return payload
}

async function loadSessionDetail(sessionId) {
  if (!sessionId) {
    return null
  }
  if (state.sessionDetails[sessionId]) {
    return state.sessionDetails[sessionId]
  }
  const payload = await getJson(`/api/v1/sessions/${encodeURIComponent(sessionId)}`)
  state.sessionDetails[sessionId] = payload
  return payload
}

async function hydrateSelections({ local = true, final = true, session = true, candidate = true } = {}) {
  const tasks = []

  if (local && state.selectedLocalSkillId) {
    tasks.push(loadSkillDetail(state.selectedLocalSkillId))
  }

  if (final && state.selectedFinalSkillId) {
    tasks.push(loadSkillDetail(state.selectedFinalSkillId))
  }

  if (session && state.selectedSessionId) {
    tasks.push(loadSessionDetail(state.selectedSessionId))
  }

  if (candidate && state.selectedCandidateJobId) {
    const job = findJobById(state.selectedCandidateJobId)
    const linkedSkill = findSkillByName(job?.skill_name || "")
    if (linkedSkill?.skill_id) {
      tasks.push(loadSkillDetail(linkedSkill.skill_id))
    }
  }

  await Promise.all(tasks.map((task) => task.catch((error) => {
    showMessage("warn", error.message || l("加载详情失败", "Failed to load details"))
  })))
}

async function refreshData({ notice = "", preserveMessage = false } = {}) {
  setLoading(true)
  if (!preserveMessage) {
    clearMessage()
  }
  try {
    const [overview, skillsPayload, sessionsPayload, validationPayload, evolve] = await Promise.all([
      getJson("/api/v1/overview"),
      getJson("/api/v1/skills?limit=500"),
      getJson("/api/v1/sessions?limit=500"),
      getJson("/api/v1/validation/jobs?limit=500"),
      getJson("/api/v1/evolve/status"),
    ])

    state.overview = overview
    state.evolve = evolve
    state.skills = sortSkills(skillsPayload.items || [])
    state.sessions = sortSessions(sessionsPayload.items || [])
    state.validationJobs = sortJobs(validationPayload.items || [])
    state.skillDetails = Object.create(null)
    state.sessionDetails = Object.create(null)

    ensureSelections()
    renderAll()
    await hydrateSelections()
    renderAll()

    if (notice) {
      showMessage("info", notice)
    }
  } catch (error) {
    showMessage("error", error.message || l("加载 dashboard 数据失败", "Failed to load dashboard data"))
  } finally {
    setLoading(false)
  }
}

function renderStaticText() {
  document.title = l("SkillClaw 技能演化看板", "SkillClaw Dashboard")
  document.documentElement.lang = state.locale === "en" ? "en" : "zh-CN"

  dom.brandTitle.textContent = l("技能演化看板", "Skill Evolution Dashboard")
  dom.brandCopy.textContent = l(
    "看你自己的技能状态、本地与共享库同步、候选验证、最终发布和会话追溯。",
    "Review your own skill status, local/shared sync, candidate validation, publication, and session traceability."
  )
  dom.navOverviewTitle.textContent = l("总览", "Overview")
  dom.navLocalTitle.textContent = l("我的技能", "My Skills")
  dom.navCandidateTitle.textContent = l("候选池", "Candidate Pool")
  dom.navFinalTitle.textContent = l("最终池", "Final Pool")
  dom.navSessionsTitle.textContent = l("会话追溯", "Sessions")
  dom.sidebarStatusTitle.textContent = l("当前状态", "Current Status")

  dom.heroKicker.textContent = l("总览首页", "Overview")
  dom.heroTitle.textContent = l("我的技能状态", "My Skill Status")
  dom.localeButton.textContent = state.locale === "en" ? "中文" : "English"
  dom.refreshButton.textContent = l("刷新数据", "Refresh")
  dom.syncButton.textContent = l("重建投影", "Rebuild Snapshot")

  dom.overviewKicker.textContent = l("总览", "Overview")
  dom.overviewTitle.textContent = l("现在这套技能更新链路处在什么状态", "Current Skill Pipeline Status")
  dom.overviewCopy.textContent = l(
    "这里汇总本地同步、候选验证、最终发布和会话来源四类状态。",
    "This page summarizes local sync, candidate validation, publication, and session sources."
  )
  dom.watchlistKicker.textContent = l("需要关注", "Watch")
  dom.watchlistTitle.textContent = l("当前值得查看的变化", "Changes Worth Checking")
  dom.scopeKicker.textContent = l("当前范围", "Scope")
  dom.scopeTitle.textContent = l("你现在看到的是哪部分数据", "What Data You Are Looking At")
  dom.eventsKicker.textContent = l("最近事件", "Recent Events")
  dom.eventsTitle.textContent = l("候选生成、验证反馈、最终决策", "Candidate Creation, Validation, and Final Decisions")
  dom.eventsCopy.textContent = l(
    "按时间倒序展示最近 12 条流程事件，方便先抓住变化再下钻。",
    "Shows the latest 12 pipeline events in reverse chronological order so you can spot changes first."
  )

  dom.localKicker.textContent = l("我的技能", "My Skills")
  dom.localTitle.textContent = l("本地技能和共享正式版是否一致", "Local Skills vs Shared Official Versions")
  dom.localCopy.textContent = l(
    "这里看你的本地技能、它们是否已同步，以及最近关联了哪些会话。",
    "Review your local skills, whether they are synced, and which sessions were recently related."
  )
  dom.localSearch.placeholder = l("搜索名称、描述或分类", "Search by name, description, or category")

  dom.candidateKicker.textContent = l("候选池", "Candidate Pool")
  dom.candidateTitle.textContent = l("哪些技能正在等待验证，哪些已经进入最终池", "Which skills are waiting for validation and which are already published")
  dom.candidateCopy.textContent = l(
    "这里只看候选技能的真实验证进度：候选内容、验证客户端、决策时间，以及它依赖哪些会话证据。",
    "This page focuses on real candidate progress: candidate content, validating clients, decision time, and supporting sessions."
  )
  dom.candidateSearch.placeholder = l("搜索技能名、候选编号或说明", "Search skill name, candidate ID, or notes")
  dom.candidateStatusAll.textContent = l("全部状态", "All Statuses")
  dom.candidateStatusPending.textContent = l("待验证", "Awaiting Validation")
  dom.candidateStatusReview.textContent = l("已有反馈", "Feedback Received")
  dom.candidateStatusPublished.textContent = l("已入最终池", "Published")
  dom.candidateStatusRejected.textContent = l("已拒绝", "Rejected")

  dom.finalKicker.textContent = l("最终池", "Final Pool")
  dom.finalTitle.textContent = l("共享技能当前正式版本和发布历史", "Shared Skill Versions and Release History")
  dom.finalCopy.textContent = l(
    "这里只显示已经进入共享库的正式技能，用来观察版本演进、发布时间和本地同步状态。",
    "This page shows only published shared skills so you can inspect version history, publish time, and local sync status."
  )
  dom.finalSearch.placeholder = l("搜索共享技能", "Search shared skills")

  dom.sessionsKicker.textContent = l("会话追溯", "Sessions")
  dom.sessionsTitle.textContent = l("我的会话，以及支撑候选技能的共享会话", "My Sessions and Shared Sessions Behind Candidate Skills")
  dom.sessionsCopy.textContent = l(
    "默认先看你的本地会话；如果需要追踪候选来源，可以切到全部会话继续查。",
    "By default you see local sessions first. Switch to all sessions if you need to trace candidate sources."
  )
  dom.sessionSearch.placeholder = l("搜索会话 ID、摘要或技能名", "Search session ID, summary, or skill name")
  dom.sessionSourceLocal.textContent = l("我的会话", "My Sessions")
  dom.sessionSourceAll.textContent = l("全部会话", "All Sessions")
  dom.sessionSourceShared.textContent = l("共享会话", "Shared Sessions")
}

function setLocale(locale) {
  state.locale = locale === "en" ? "en" : "zh"
  try {
    window.localStorage.setItem(LOCALE_STORAGE_KEY, state.locale)
  } catch {}
  renderAll()
}

function renderAll() {
  renderStaticText()
  renderNav()
  renderSidebarStatus()
  renderOverview()
  renderLocalPage()
  renderCandidatePage()
  renderFinalPage()
  renderSessionsPage()
}

function renderNav() {
  for (const button of dom.navButtons) {
    button.classList.toggle("active", button.dataset.view === state.activeView)
  }
  for (const panel of dom.viewPanels) {
    panel.classList.toggle("active", panel.dataset.viewPanel === state.activeView)
  }

  const counts = candidateCounts()
  dom.navOverviewMeta.textContent = l("{pending} 待验证 · {published} 已发布", "{pending} awaiting · {published} published", {
    pending: number(counts.pending),
    published: number(counts.published),
  })
  dom.navLocalMeta.textContent = l("{count} 个本地技能", "{count} local skills", {
    count: number(localSkillItems().length),
  })
  dom.navCandidateMeta.textContent = l("{count} 条候选", "{count} candidates", {
    count: number(counts.total),
  })
  dom.navFinalMeta.textContent = l("{count} 个共享技能", "{count} shared skills", {
    count: number(finalSkillItems().length),
  })
  dom.navSessionsMeta.textContent = l("{count} 个我的会话", "{count} my sessions", {
    count: number(mySessionItems().length),
  })
}

function renderSidebarStatus() {
  const sync = localSyncCounts()
  const counts = candidateCounts()
  dom.sidebarStatus.innerHTML = [
    renderContextCard(l("我的技能", "My Skills"), [
      [l("总数", "Total"), l("{count} 个", "{count}", { count: number(localSkillItems().length) })],
      [l("已同步", "Synced"), l("{count} 个", "{count}", { count: number(sync.synced) })],
    ]),
    renderContextCard(l("候选", "Candidates"), [
      [l("待验证", "Awaiting"), l("{count} 条", "{count}", { count: number(counts.pending) })],
      [l("待决策", "Awaiting Decision"), l("{count} 条", "{count}", { count: number(counts.review) })],
    ]),
    renderContextCard(l("共享状态", "Sharing"), [
      [l("共享库", "Shared Pool"), sharingEnabled() ? l("已启用", "Enabled") : l("未启用", "Disabled")],
      [l("演化服务", "Evolve"), evolveHealthLabel()],
    ]),
  ].join("")
}

function renderOverview() {
  const counts = candidateCounts()
  const sync = localSyncCounts()

  dom.overviewMetrics.innerHTML = [
    renderMetricCard(l("我的技能", "My Skills"), localSkillItems().length, l("当前机器上可直接使用的技能数量。", "Number of skills currently available on this machine.")),
    renderMetricCard(l("已同步", "Synced"), sync.synced, l("本地内容已经和共享正式版一致。", "Local content matches the shared official version.")),
    renderMetricCard(l("待处理同步", "Needs Sync"), sync.drift + sync.localOnly, l("本地与共享库仍有差异，或尚未发布到共享库。", "Local content still differs from the shared pool or has not been published.")),
    renderMetricCard(l("待验证候选", "Pending Candidates"), counts.pending + counts.review, l("已经进入候选池，但还没有完全走完验证和决策。", "Candidates have entered the pool but have not finished validation and decision making.")),
    renderMetricCard(l("已入最终池", "Published"), counts.published, l("通过验证并完成发布决策的候选。", "Candidates that passed validation and were published.")),
    renderMetricCard(l("我的会话", "My Sessions"), mySessionItems().length, l("当前本地采集到的个人会话数。", "Number of personal sessions currently captured locally.")),
  ].join("")

  const watchlist = buildWatchlistItems()
  dom.overviewWatchlist.innerHTML = watchlist.length
    ? watchlist.map(renderJumpCard).join("")
    : `<div class="empty-state">${escapeHtml(l("当前没有明显堵点，整体状态比较稳定。", "No obvious blockers right now. The overall pipeline looks stable."))}</div>`

  const contextCards = [
    renderContextCard(l("同步情况", "Sync Status"), [
      [l("已同步", "Synced"), l("{count} 个技能", "{count} skills", { count: number(sync.synced) })],
      [l("待更新", "Out of Date"), l("{count} 个技能", "{count} skills", { count: number(sync.drift) })],
      [l("仅本地", "Local Only"), l("{count} 个技能", "{count} skills", { count: number(sync.localOnly) })],
    ]),
    renderContextCard(l("候选进度", "Candidate Progress"), [
      [l("待验证", "Awaiting"), l("{count} 条", "{count}", { count: number(counts.pending) })],
      [l("待决策", "Awaiting Decision"), l("{count} 条", "{count}", { count: number(counts.review) })],
      [l("已发布", "Published"), l("{count} 条", "{count}", { count: number(counts.published) })],
    ]),
    renderContextCard(l("当前范围", "Scope"), [
      [l("共享库", "Shared Pool"), sharingEnabled() ? l("已启用", "Enabled") : l("未启用", "Disabled")],
      [l("共享技能", "Shared Skills"), l("{count} 个", "{count}", { count: number(finalSkillItems().length) })],
      [l("最近 24 小时变化", "Changes in 24h"), l("{count} 条", "{count}", { count: number(recentEventCount(24)) })],
    ]),
  ]

  if (warnings().length) {
    contextCards.push(`
      <article class="context-card tone-pending">
        <div class="headline-row">
          <div>
            <p class="kicker">${escapeHtml(l("告警", "Warnings"))}</p>
            <h4>${escapeHtml(l("采集或同步时发现了问题", "Issues Found During Collection or Sync"))}</h4>
          </div>
          ${badge(`${number(warnings().length)} 条`, "pending")}
        </div>
        <ol class="rule-list">
          ${warnings().map((item) => `<li>${escapeHtml(String(item || ""))}</li>`).join("")}
        </ol>
      </article>
    `)
  }

  dom.overviewContext.innerHTML = contextCards.join("")

  const events = collectPipelineEvents()
  dom.overviewEvents.innerHTML = events.length
    ? events.map(renderEventCard).join("")
    : `<div class="empty-state">${escapeHtml(l("最近还没有可展示的候选、验证或发布事件。", "There are no candidate, validation, or publish events to show yet."))}</div>`
}

function buildWatchlistItems() {
  const items = []
  const sync = localSyncCounts()
  const counts = candidateCounts()
  const publishedRecent = candidateItems()
    .filter((job) => candidateStatusKey(job.status) === "published")
    .slice(0, 2)
  const pendingJob = candidateItems().find((job) => candidateStatusKey(job.status) === "pending")
  const reviewJob = candidateItems().find((job) => candidateStatusKey(job.status) === "review")
  const driftSkill = localSkillItems().find((skill) => compareLocalAndRemote(skill).tone === "pending")

  if (driftSkill) {
    items.push({
      view: "local",
      title: l("{name} 和共享正式版存在差异", "{name} differs from the shared official version", { name: driftSkill.name }),
      copy: l("当前状态：{status}", "Current status: {status}", { status: compareLocalAndRemote(driftSkill).label }),
      dataset: `data-select-local-skill="${escapeHtml(driftSkill.skill_id)}"`,
      tone: "pending",
    })
  }

  if (pendingJob) {
    items.push({
      view: "candidate",
      title: l("{count} 条候选仍在等待验证", "{count} candidates are still awaiting validation", { count: number(counts.pending) }),
      copy: l("{name} 还在等待更多验证反馈。", "{name} is still waiting for more validation feedback.", { name: pendingJob.skill_name }),
      dataset: `data-select-candidate="${escapeHtml(pendingJob.job_id)}"`,
      tone: "pending",
    })
  }

  if (reviewJob) {
    items.push({
      view: "candidate",
      title: l("{count} 条候选已有反馈但还没完成决策", "{count} candidates have feedback but no final decision yet", { count: number(counts.review) }),
      copy: l("{name} 已收到结果，正在等待最终结论。", "{name} already has results and is waiting for the final decision.", { name: reviewJob.skill_name }),
      dataset: `data-select-candidate="${escapeHtml(reviewJob.job_id)}"`,
      tone: "pending",
    })
  }

  for (const job of publishedRecent) {
    const linkedSkill = findSkillByName(job.skill_name)
    items.push({
      view: linkedSkill ? "final" : "candidate",
      title: l("{name} 最近进入最终池", "{name} was recently published", { name: job.skill_name }),
      copy: l("发布时间 {time}。", "Published at {time}.", { time: formatStamp(jobDecision(job).decided_at) }),
      dataset: linkedSkill
        ? `data-select-final-skill="${escapeHtml(linkedSkill.skill_id)}"`
        : `data-select-candidate="${escapeHtml(job.job_id)}"`,
      tone: "published",
    })
  }

  if (mySessionItems().length) {
    const latestSession = mySessionItems()[0]
    items.push({
      view: "sessions",
      title: l("最近一条本地会话", "Most Recent Local Session"),
      copy: clip(latestSession.prompt_preview || latestSession.response_preview || l("查看最近一次会话。", "View the latest session."), 80),
      dataset: `data-select-session="${escapeHtml(latestSession.session_id)}"`,
      tone: "neutral",
    })
  }

  return items.slice(0, 5)
}

function renderJumpCard(item) {
  return `
    <button
      class="jump-card"
      data-open-view="${escapeHtml(item.view)}"
      ${item.dataset || ""}
      type="button"
    >
      <div class="headline-row">
        <div>
          <strong>${escapeHtml(item.title)}</strong>
          <p class="soft-copy">${escapeHtml(item.copy)}</p>
        </div>
        ${badge(l("查看", "Open"), item.tone || "neutral")}
      </div>
    </button>
  `
}

function renderMetricCard(label, value, note) {
  return `
    <article class="metric-card">
      <p class="metric-label">${escapeHtml(label)}</p>
      <div class="metric-value">${escapeHtml(String(value))}</div>
      <p class="metric-note">${escapeHtml(note)}</p>
    </article>
  `
}

function renderContextCard(title, rows) {
  return `
    <article class="context-card">
      <div class="headline-row">
        <div>
          <p class="kicker">${escapeHtml(title)}</p>
        </div>
      </div>
      <div class="kv-list">
        ${rows.map(([key, value]) => `
          <div class="context-row">
            <span class="kv-key">${escapeHtml(String(key))}</span>
            <span class="kv-value">${escapeHtml(String(value))}</span>
          </div>
        `).join("")}
      </div>
    </article>
  `
}

function renderEventCard(event) {
  return `
    <article class="event-card tone-${escapeHtml(event.tone || "neutral")}">
      <div class="event-row">
        <div>
          <p class="kicker">${escapeHtml(eventTypeLabel(event.type))}</p>
          <h4>${escapeHtml(event.title)}</h4>
        </div>
        ${badge(formatStamp(event.timestamp), event.tone || "neutral")}
      </div>
      <p class="soft-copy">${escapeHtml(clip(event.copy || l("无额外说明。", "No additional details."), 180))}</p>
      ${event.jobId
        ? `
          <div class="action-row">
            <button class="ghost" data-open-view="candidate" data-select-candidate="${escapeHtml(event.jobId)}" type="button">
              ${escapeHtml(l("查看候选详情", "View Candidate"))}
            </button>
          </div>
        `
        : ""}
    </article>
  `
}

function renderLocalPage() {
  const items = filteredLocalSkills()
  const detail = selectedLocalSkillDetail()

  dom.localList.innerHTML = items.length
    ? items.map(renderLocalSkillCard).join("")
    : `<div class="empty-state">${escapeHtml(l("当前筛选条件下没有本地技能。", "No local skills match the current filter."))}</div>`

  dom.localDetail.innerHTML = detail
    ? renderLocalSkillDetail(detail)
    : `<div class="empty-state">${escapeHtml(l("选择一个本地技能，查看同步状态、版本链和相关会话。", "Select a local skill to inspect sync status, version history, and related sessions."))}</div>`
}

function renderLocalSkillCard(skill) {
  const active = String(skill.skill_id || "") === String(state.selectedLocalSkillId || "")
  const sync = compareLocalAndRemote(skill)
  const candidateCount = candidateRecordCount(skill.name)
  return `
    <article class="record-card ${active ? "active" : ""}" data-select-local-skill="${escapeHtml(skill.skill_id)}">
      <div class="card-head">
        <div>
          <p class="card-kicker">${escapeHtml(categoryLabel(skill.category))}</p>
          <h4>${escapeHtml(skill.name || l("未命名技能", "Unnamed Skill"))}</h4>
        </div>
        ${badge(sync.label, sync.tone)}
      </div>
      <p class="card-copy">${escapeHtml(clip(skill.description || l("这个技能还没有描述。", "This skill does not have a description yet."), 110))}</p>
      <div class="chip-row">
        ${tag(l("关联会话 {count}", "Sessions {count}", { count: number(skill.session_count || 0) }))}
        ${tag(l("候选记录 {count}", "Candidates {count}", { count: number(candidateCount) }))}
        ${skill.has_remote ? tag(l("共享正式版 {version}", "Shared Version {version}", { version: sharedVersionLabel(skill) })) : tag(l("尚未进入共享库", "Not in Shared Pool"))}
      </div>
    </article>
  `
}

function renderLocalSkillDetail(skill) {
  const sync = compareLocalAndRemote(skill)
  const candidateCount = candidateRecordCount(skill.name)
  return `
    <div class="stack">
      <section class="detail-card">
        <div class="detail-head">
          <div>
            <p class="kicker">${escapeHtml(l("本地技能", "Local Skill"))}</p>
            <h3>${escapeHtml(skill.name || l("未命名技能", "Unnamed Skill"))}</h3>
            <p class="soft-copy">${escapeHtml(skill.description || l("这个技能还没有描述。", "This skill does not have a description yet."))}</p>
          </div>
          <div class="chip-row">
            ${badge(sync.label, sync.tone)}
            ${badge(skill.has_remote ? l("本地和共享库都有", "Local + Shared") : l("仅本地", "Local Only"), "neutral")}
          </div>
        </div>
        <div class="mini-grid">
        ${renderMiniCard(l("当前状态", "Current Status"), sync.label, l("本地内容和共享正式版现在是什么关系。", "How the local content relates to the shared official version."))}
        ${renderMiniCard(l("共享正式版", "Shared Official Version"), sharedVersionLabel(skill), skill.has_remote ? l("共享库当前正式版本。", "Current official version in the shared pool.") : l("这个技能还没有进入共享库。", "This skill has not entered the shared pool yet."))}
        ${renderMiniCard(l("候选记录", "Candidate Records"), number(candidateCount), candidateCount ? l("这个技能最近进入过多少次候选验证流程。", "How many candidate validation flows this skill entered recently.") : l("目前没有候选记录。", "There are no candidate records yet."))}
        ${renderMiniCard(l("关联会话", "Related Sessions"), number(skill.session_count || 0), l("最近有多少条会话和这个技能有关。", "How many recent sessions are related to this skill."))}
      </div>
      </section>
      ${renderVersionCompare(skill, {
        scope: "local",
        title: l("本地版本与共享正式版对比", "Compare Local vs Shared"),
        copy: l("左边默认是本地当前版，右边可以切换共享历史版本，用来判断内容是否已经同步。", "The left side shows the current local version. Switch the right side to shared history versions to compare sync state."),
        includeLocal: true,
      })}
      ${skill.has_remote
        ? renderVersionTimeline(skill, {
          title: l("共享版本历史", "Shared Version History"),
          copy: l("这个技能在共享库中的版本演进。", "How this skill evolved inside the shared pool."),
        })
        : renderSharingNotConnectedNotice(l("这个技能还没有进入共享正式版，所以这里没有版本历史。", "This skill is not in the shared official pool yet, so there is no version history here."))}
      ${renderCandidateLinks(skill.name)}
      ${renderRelatedSessions(skill.related_sessions, { emptyTitle: l("最近没有观测到相关会话", "No related sessions were observed recently") })}
    </div>
  `
}

function renderMiniCard(label, value, note) {
  return `
    <article class="mini-card">
      <p class="metric-label">${escapeHtml(label)}</p>
      <div class="mini-value">${escapeHtml(String(value))}</div>
      <p class="soft-copy">${escapeHtml(String(note || ""))}</p>
    </article>
  `
}

function renderCandidateLinks(skillName) {
  const jobs = jobsForSkill(skillName)
  if (!jobs.length) {
    return `
      <section class="detail-card">
        <div class="headline-row">
          <div>
            <p class="kicker">${escapeHtml(l("候选记录", "Candidate Records"))}</p>
            <h4>${escapeHtml(l("这个技能目前没有候选记录", "This skill does not have candidate records yet"))}</h4>
          </div>
        </div>
      </section>
    `
  }
  return `
    <section class="detail-card">
      <div class="headline-row">
        <div>
          <p class="kicker">${escapeHtml(l("候选记录", "Candidate Records"))}</p>
          <h4>${escapeHtml(l("这个技能最近进入过哪些验证流程", "Which validation flows this skill recently entered"))}</h4>
        </div>
        ${badge(`${number(jobs.length)} 条`, "neutral")}
      </div>
      <div class="stack">
        ${jobs.map((job) => `
          <button class="jump-card" data-open-view="candidate" data-select-candidate="${escapeHtml(job.job_id)}" type="button">
            <div class="headline-row">
              <div>
                <strong>${escapeHtml(job.job_id)}</strong>
                <p class="soft-copy">${escapeHtml(actionLabel(job.proposed_action))} · ${escapeHtml(candidateStatusLabel(job.status))}</p>
              </div>
              ${badge(formatStamp(job.created_at), toneForStatus(job.status))}
            </div>
          </button>
        `).join("")}
      </div>
    </section>
  `
}

function renderRelatedSessions(items, { emptyTitle = "" } = {}) {
  const sessions = Array.isArray(items) ? items : []
  const resolvedEmptyTitle = emptyTitle || l("没有相关会话", "No related sessions")
  if (!sessions.length) {
    return `
      <section class="detail-card">
        <div class="headline-row">
          <div>
            <p class="kicker">${escapeHtml(l("会话追溯", "Sessions"))}</p>
            <h4>${escapeHtml(resolvedEmptyTitle)}</h4>
          </div>
        </div>
      </section>
    `
  }
  return `
    <section class="detail-card">
      <div class="headline-row">
        <div>
          <p class="kicker">${escapeHtml(l("会话追溯", "Sessions"))}</p>
          <h4>${escapeHtml(l("最近关联到这个技能的会话", "Recent sessions related to this skill"))}</h4>
        </div>
        ${badge(l("{count} 条", "{count}", { count: number(sessions.length) }), "neutral")}
      </div>
      <div class="stack">
        ${sessions.slice(0, 8).map((item) => `
          <button class="jump-card" data-open-view="sessions" data-select-session="${escapeHtml(item.session_id)}" type="button">
            <div class="headline-row">
              <div>
                <strong>${escapeHtml(item.session_id || l("未知会话", "Unknown Session"))}</strong>
                <p class="soft-copy">${escapeHtml(clip(item.prompt_preview || item.response_preview || l("没有摘要。", "No summary."), 110))}</p>
              </div>
              ${badge(formatStamp(item.timestamp), "neutral")}
            </div>
          </button>
        `).join("")}
      </div>
    </section>
  `
}

function renderCandidatePage() {
  const items = filteredCandidateJobs()
  const detail = selectedCandidateJob()

  dom.candidateList.innerHTML = items.length
    ? items.map(renderCandidateCard).join("")
    : renderCandidateEmptyState()

  dom.candidateDetail.innerHTML = detail
    ? renderCandidateDetail(detail)
    : renderCandidateDetailEmptyState()
}

function renderCandidateCard(job) {
  const active = String(job.job_id || "") === String(state.selectedCandidateJobId || "")
  const tone = toneForStatus(job.status)
  const core = jobPayload(job)
  const results = jobResults(job)
  const dispatch = validationDispatchSummary(job)
  return `
    <article class="record-card ${active ? "active" : ""}" data-select-candidate="${escapeHtml(job.job_id)}">
      <div class="card-head">
        <div>
          <p class="card-kicker">${escapeHtml(job.job_id || l("未知候选编号", "Unknown Candidate ID"))}</p>
          <h4>${escapeHtml(job.skill_name || l("未命名技能", "Unnamed Skill"))}</h4>
        </div>
        ${badge(candidateStatusLabel(job.status), tone)}
      </div>
      <p class="card-copy">${escapeHtml(clip(core.rationale || l("这个候选没有额外说明。", "This candidate has no additional notes."), 110))}</p>
      <div class="chip-row">
        ${tag(actionLabel(job.proposed_action))}
        ${tag(l("进入池 {time}", "Entered Pool {time}", { time: formatStamp(job.created_at) }))}
        ${tag(l("已回 {count}", "Returned {count}", { count: number(results.length) }))}
        ${dispatch.pendingResults > 0 ? tag(l("还差 {count} 条结果", "{count} more results needed", { count: number(dispatch.pendingResults) })) : tag(l("结果数已满足", "Enough results received"))}
        ${tag(l("会话 {count}", "Sessions {count}", { count: number(jobSessionIds(job).length) }))}
      </div>
    </article>
  `
}

function renderCandidateDetail(job) {
  const tone = toneForStatus(job.status)
  const core = jobPayload(job)
  const results = jobResults(job)
  const decision = jobDecision(job)
  const linkedSkill = findSkillByName(job.skill_name)
  const linkedDetail = linkedSkill ? state.skillDetails[linkedSkill.skill_id] : null
  const candidateSkill = core.candidate_skill || {}
  const currentSkill = core.current_skill || {}
  const sessionEvidence = Array.isArray(core.session_evidence) ? core.session_evidence : []

  return `
    <div class="stack">
      <section class="detail-card tone-${escapeHtml(tone)}">
        <div class="detail-head">
          <div>
            <p class="kicker">${escapeHtml(l("候选技能", "Candidate Skill"))}</p>
            <h3>${escapeHtml(job.skill_name || l("未命名技能", "Unnamed Skill"))}</h3>
            <p class="soft-copy">${escapeHtml(core.rationale || l("这个候选没有额外说明。", "This candidate has no additional notes."))}</p>
          </div>
          <div class="chip-row">
            ${badge(candidateStatusLabel(job.status), tone)}
            ${badge(actionLabel(job.proposed_action), "neutral")}
          </div>
        </div>
        <div class="mini-grid">
          ${renderMiniCard(l("进入候选池", "Entered Candidate Pool"), formatStamp(job.created_at), l("这条候选开始进入验证流程的时间。", "When this candidate entered the validation pipeline."))}
          ${renderMiniCard(l("验证结果", "Validation Results"), `${number(job.accepted_count || 0)} / ${number(job.result_count || results.length)}`, l("通过数 / 总反馈数", "Accepted / total feedback"))}
          ${renderMiniCard(l("平均分", "Average Score"), formatScore(job.mean_score), l("当前验证结果的平均分", "Average score across current validation feedback"))}
          ${renderMiniCard(l("最终决策", "Final Decision"), candidateStatusLabel(decision.status || job.status), decision?.decided_at ? l("决策时间 {time}", "Decided at {time}", { time: formatStamp(decision.decided_at) }) : l("还没有进入最终决策", "No final decision yet"))}
        </div>
      </section>
      ${renderCandidateStages(job, linkedDetail)}
      ${renderDispatchSection(job)}
      <section class="detail-card">
        <div class="headline-row">
          <div>
            <p class="kicker">${escapeHtml(l("验证阈值", "Validation Thresholds"))}</p>
            <h4>${escapeHtml(l("这条候选需要满足什么条件才能进入最终池", "Conditions required for this candidate to be published"))}</h4>
          </div>
        </div>
        <div class="mini-grid">
          ${renderMiniCard(l("最少结果数", "Min Results"), number(core.min_results || 0), l("需要收集的最少验证结果", "Minimum validation results required"))}
          ${renderMiniCard(l("最少通过数", "Min Approvals"), number(core.min_approvals || 0), l("至少要有多少个验证客户端给出通过结果。", "Minimum number of validating clients that must approve"))}
          ${renderMiniCard(l("最低均分", "Min Average Score"), formatScore(core.min_score), l("平均分阈值", "Average score threshold"))}
          ${renderMiniCard(l("最大拒绝数", "Max Rejections"), number(core.max_rejections || 0), l("达到这个数就会被拒绝", "The candidate is rejected when this threshold is reached"))}
        </div>
      </section>
      ${renderCandidateDocuments(candidateSkill, currentSkill, linkedDetail)}
      ${linkedDetail
        ? renderVersionTimeline(linkedDetail, {
          title: l("若发布成功，将进入这条版本链", "This is the version chain the candidate will join if published"),
          copy: l("这里展示的是该技能当前在最终池中的已知历史版本。", "This shows the known published history of the skill in the final pool."),
        })
        : renderCandidateVersionNotice(job)}
      ${renderValidatorSection(results)}
      ${renderSessionEvidenceSection(sessionEvidence, jobSessionIds(job))}
      ${renderCandidateOutcomeSection(job, linkedSkill, decision)}
    </div>
  `
}

function renderCandidateStages(job, linkedDetail) {
  const results = jobResults(job)
  const decision = jobDecision(job)
  const status = candidateStatusKey(job.status)
  const publishVersion = linkedDetail && candidateStatusKey(decision.status || job.status) === "published"
    ? l("最终池当前版本 v{version}", "Current final version v{version}", { version: number(linkedDetail.current_version || 0) })
    : l("等待正式入池", "Waiting to be published")

  const stages = [
    {
      title: l("候选生成", "Candidate Created"),
      state: "done",
      stamp: job.created_at,
      note: l("{action} · 候选编号 {jobId}", "{action} · candidate ID {jobId}", { action: actionLabel(job.proposed_action), jobId: job.job_id }),
    },
    {
      title: l("验证反馈", "Validation Feedback"),
      state: results.length ? (status === "pending" ? "current" : "done") : "current",
      stamp: results.length ? results[results.length - 1].created_at : "",
      note: results.length
        ? l("已收到 {count} 条结果", "Received {count} results", { count: number(results.length) })
        : l("还没有任何客户端提交验证结果", "No client has submitted validation feedback yet"),
    },
    {
      title: l("最终决策", "Final Decision"),
      state: decision?.status
        ? (candidateStatusKey(decision.status) === "rejected" ? "blocked" : "done")
        : (status === "review" ? "current" : "pending"),
      stamp: decision?.decided_at || "",
      note: decision?.status
        ? candidateStatusLabel(decision.status)
        : l("等待系统根据阈值汇总验证结果", "Waiting for the system to aggregate validation results"),
    },
    {
      title: l("进入最终池", "Published to Final Pool"),
      state: status === "published" ? "done" : status === "rejected" ? "blocked" : "pending",
      stamp: candidateStatusKey(decision?.status) === "published" ? (decision.decided_at || "") : "",
      note: status === "published"
        ? publishVersion
        : status === "rejected"
          ? l("这条候选没有进入最终池", "This candidate was not published")
          : l("还没有正式发布到共享库", "Not yet published to the shared pool"),
    },
  ]

  return `
    <section class="detail-card">
      <div class="headline-row">
        <div>
          <p class="kicker">${escapeHtml(l("阶段追踪", "Stage Tracking"))}</p>
          <h4>${escapeHtml(l("这条候选是如何沿着更新流程往前走的", "How this candidate progressed through the pipeline"))}</h4>
        </div>
      </div>
      <div class="stage-grid">
        ${stages.map((stage) => `
          <article class="stage-card ${escapeHtml(stage.state)} ${stage.state === "current" ? "current" : ""}">
            <p class="status-caption">${escapeHtml(stageStateLabel(stage.state))}</p>
            <div class="stage-title">${escapeHtml(stage.title)}</div>
            <p class="stage-note">${escapeHtml(stage.stamp ? formatStamp(stage.stamp) : l("无时间", "No time"))}</p>
            <p class="stage-note">${escapeHtml(stage.note)}</p>
          </article>
        `).join("")}
      </div>
    </section>
  `
}

function renderCandidateDocuments(candidateSkill, currentSkill, linkedDetail) {
  const candidateDoc = skillDocumentPreview(candidateSkill)
  const currentDoc = skillDocumentPreview(currentSkill)
    || String(linkedDetail?.remote_skill_md || linkedDetail?.remote_content || "").trim()
  return `
    <section class="detail-card">
      <div class="headline-row">
        <div>
          <p class="kicker">${escapeHtml(l("内容对照", "Content Comparison"))}</p>
          <h4>${escapeHtml(l("候选草案和当前最终版有什么区别", "How the candidate draft differs from the current official version"))}</h4>
        </div>
      </div>
      <div class="doc-grid">
        ${renderDocCard(l("候选草案", "Candidate Draft"), candidateDoc, {
          meta: [
            badge(categoryLabel(candidateSkill?.category || "candidate"), "neutral"),
            candidateSkill?.description ? tag(candidateSkill.description) : "",
          ],
        })}
        ${renderDocCard(l("当前最终版", "Current Official Version"), currentDoc, {
          meta: [
            badge(currentSkill?.name || linkedDetail?.name || l("共享正式版", "Shared Official Version"), "published"),
            linkedDetail?.current_version ? tag(`v${number(linkedDetail.current_version)}`) : "",
          ],
          emptyText: l("当前没有拿到对应的最终池文档快照。", "No official snapshot is available for this final-pool version."),
        })}
      </div>
    </section>
  `
}

function renderDispatchSection(job) {
  const core = jobPayload(job)
  const results = jobResults(job)
  const dispatch = validationDispatchSummary(job)
  return `
    <section class="detail-card">
      <div class="headline-row">
        <div>
          <p class="kicker">${escapeHtml(l("分发与响应", "Dispatch and Responses"))}</p>
          <h4>${escapeHtml(l("这条候选是什么时候进入验证池，又由哪些客户端返回结果", "When this candidate entered validation and which clients responded"))}</h4>
        </div>
      </div>
      <div class="mini-grid">
        ${renderMiniCard(l("进入验证池", "Entered Validation"), formatStamp(dispatch.dispatchAt), l("这条候选开始对验证客户端可见的时间。", "When this candidate became visible to validating clients."))}
        ${renderMiniCard(l("分发方式", "Dispatch Mode"), dispatch.dispatchLabel, l("当前是开放领取，不是后台预先分配给某一台机器。", "This is open pickup, not pre-assigned to a specific machine."))}
        ${renderMiniCard(l("仍缺结果", "Results Missing"), number(dispatch.pendingResults), dispatch.pendingResults > 0 ? l("还需要更多客户端返回结果", "More client results are still needed") : l("结果数已达到最低阈值", "Minimum result count reached"))}
        ${renderMiniCard(l("仍缺通过", "Approvals Missing"), number(dispatch.pendingApprovals), dispatch.pendingApprovals > 0 ? l("还需要更多接受结果", "More approvals are still needed") : l("通过数已达到最低阈值", "Minimum approval count reached"))}
      </div>
      <div class="stack">
        ${results.length
          ? results
            .slice()
            .sort((left, right) => parseTime(left.created_at) - parseTime(right.created_at))
            .map((result) => `
              <article class="validator-card tone-${result.accepted === true ? "published" : "rejected"}">
                <div class="validator-head">
                  <div>
                    <h4>${escapeHtml(result.user_alias || l("未知客户端", "Unknown Client"))}</h4>
                    <p class="soft-copy">${escapeHtml(formatStamp(result.created_at))}</p>
                  </div>
                  ${badge(result.accepted === true ? l("返回通过", "Approved") : l("返回拒绝", "Rejected"), result.accepted === true ? "published" : "rejected")}
                </div>
                <p class="soft-copy">${escapeHtml(validatorModeLabel(result.validator_mode))} · ${escapeHtml(l("分数", "Score"))} ${escapeHtml(formatScore(result.score))}</p>
              </article>
            `).join("")
          : `
            <div class="empty-state">
              ${escapeHtml(l("这条候选已经在 {time} 进入开放验证池，但当前还没有任何客户端返回结果。", "This candidate entered the open validation pool at {time}, but no client has responded yet.", { time: formatStamp(dispatch.dispatchAt) }))}
            </div>
          `}
      </div>
      ${core.source
        ? `<p class="soft-copy">${escapeHtml(l("候选来源：{source}", "Candidate source: {source}", { source: candidateSourceLabel(core.source) }))}</p>`
        : ""}
    </section>
  `
}

function renderCandidateVersionNotice(job) {
  return `
    <section class="detail-card tone-pending">
      <div class="headline-row">
        <div>
          <p class="kicker">${escapeHtml(l("版本链", "Version History"))}</p>
          <h4>${escapeHtml(l("当前还没有可关联的最终池版本链", "There is no linked final-pool version history yet"))}</h4>
        </div>
      </div>
      <p class="soft-copy">
        ${escapeHtml(l("{name} 还没有在当前投影里对应到共享最终池中的正式技能。若后续发布成功，版本链会出现在这里。", "{name} is not yet linked to an official skill in the current final pool projection. If it is published later, the version history will appear here.", {
          name: job.skill_name || l("这个候选", "This candidate"),
        }))}
      </p>
    </section>
  `
}

function renderDocCard(title, document, { meta = [], emptyText = l("没有可展示的文档。", "No document is available to display.") } = {}) {
  return `
    <article class="doc-card">
      <div class="doc-head">
        <div>
          <p class="kicker">${escapeHtml(title)}</p>
        </div>
        <div class="chip-row">${meta.join("")}</div>
      </div>
      ${String(document || "").trim()
        ? `<pre>${escapeHtml(String(document).trim())}</pre>`
        : `<div class="empty-state">${escapeHtml(emptyText)}</div>`}
    </article>
  `
}

function renderValidatorSection(results) {
  if (!results.length) {
    return `
      <section class="detail-card">
        <div class="headline-row">
          <div>
            <p class="kicker">${escapeHtml(l("验证反馈", "Validation Feedback"))}</p>
            <h4>${escapeHtml(l("还没有任何客户端提交结果", "No client has submitted results yet"))}</h4>
          </div>
        </div>
      </section>
    `
  }

  return `
    <section class="detail-card">
      <div class="headline-row">
        <div>
          <p class="kicker">${escapeHtml(l("验证反馈", "Validation Feedback"))}</p>
          <h4>${escapeHtml(l("哪些客户端验证过这个候选，给了什么分", "Which clients validated this candidate and what scores they gave"))}</h4>
        </div>
        ${badge(`${number(results.length)} 条`, "neutral")}
      </div>
      <div class="validator-grid">
        ${results.map((result) => {
          const accepted = result.accepted === true
          return `
            <article class="validator-card tone-${accepted ? "published" : "rejected"}">
              <div class="validator-head">
                <div>
                  <h4>${escapeHtml(result.user_alias || l("未知客户端", "Unknown Client"))}</h4>
                  <p class="soft-copy">${escapeHtml(validatorModeLabel(result.validator_mode))}</p>
                </div>
                ${badge(accepted ? l("通过", "Accepted") : l("拒绝", "Rejected"), accepted ? "published" : "rejected")}
              </div>
              <div class="chip-row">
                ${tag(l("分数 {score}", "Score {score}", { score: formatScore(result.score) }))}
                ${tag(formatStamp(result.created_at))}
              </div>
              <p class="soft-copy">${escapeHtml(result.notes || result.reason || l("没有备注。", "No notes."))}</p>
            </article>
          `
        }).join("")}
      </div>
    </section>
  `
}

function renderSessionEvidenceSection(evidence, sessionIds) {
  const rows = Array.isArray(evidence) ? evidence : []
  const fallbackRows = rows.length
    ? rows
    : sessionIds.map((sessionId) => ({
      session_id: sessionId,
      summary: l("这条候选引用了该会话，但没有额外摘要。", "This candidate references the session, but no additional summary is available."),
    }))

  if (!fallbackRows.length) {
    return `
      <section class="detail-card">
        <div class="headline-row">
          <div>
            <p class="kicker">${escapeHtml(l("会话证据", "Session Evidence"))}</p>
            <h4>${escapeHtml(l("这条候选没有记录会话来源", "This candidate has no recorded session source"))}</h4>
          </div>
        </div>
      </section>
    `
  }

  return `
    <section class="detail-card">
      <div class="headline-row">
        <div>
          <p class="kicker">${escapeHtml(l("会话证据", "Session Evidence"))}</p>
          <h4>${escapeHtml(l("哪些会话支撑了这条候选", "Which sessions support this candidate"))}</h4>
        </div>
        ${badge(`${number(fallbackRows.length)} 条`, "neutral")}
      </div>
      <div class="stack">
        ${fallbackRows.map((item) => `
          <button class="jump-card" data-open-view="sessions" data-select-session="${escapeHtml(item.session_id || "")}" type="button">
            <div class="headline-row">
              <div>
                <strong>${escapeHtml(item.session_id || l("未知会话", "Unknown Session"))}</strong>
                <p class="soft-copy">${escapeHtml(clip(item.summary || l("没有额外摘要。", "No additional summary."), 120))}</p>
              </div>
              <div class="chip-row">
                ${item.judge_overall_score != null ? badge(l("总评 {score}", "Overall {score}", { score: formatScore(item.judge_overall_score) }), "neutral") : ""}
                ${item.avg_prm != null ? tag(l("自动评分 {score}", "Auto Score {score}", { score: formatScore(item.avg_prm) })) : ""}
              </div>
            </div>
          </button>
        `).join("")}
      </div>
    </section>
  `
}

function renderCandidateOutcomeSection(job, linkedSkill, decision) {
  const status = candidateStatusKey(decision?.status || job.status)
  if (status === "published") {
    return `
      <section class="detail-card tone-published">
        <div class="headline-row">
          <div>
            <p class="kicker">${escapeHtml(l("结果", "Result"))}</p>
            <h4>${escapeHtml(l("这条候选已经进入最终池", "This candidate has been published"))}</h4>
          </div>
          ${badge(l("已发布", "Published"), "published")}
        </div>
        <p class="soft-copy">
          ${escapeHtml(decision.reason || l("候选满足验证阈值后进入最终池。", "The candidate met the validation thresholds and was published."))}
        </p>
        ${linkedSkill
          ? `
            <div class="action-row">
              <button class="ghost" data-open-view="final" data-select-final-skill="${escapeHtml(linkedSkill.skill_id)}" type="button">
                ${escapeHtml(l("查看最终池版本", "View Published Skill"))}
              </button>
            </div>
          `
          : ""}
      </section>
    `
  }
  if (status === "rejected") {
    return `
      <section class="detail-card tone-rejected">
        <div class="headline-row">
          <div>
            <p class="kicker">${escapeHtml(l("结果", "Result"))}</p>
            <h4>${escapeHtml(l("这条候选没有进入最终池", "This candidate was not published"))}</h4>
          </div>
          ${badge(l("已拒绝", "Rejected"), "rejected")}
        </div>
        <p class="soft-copy">${escapeHtml(decision.reason || l("当前验证结果未满足发布条件。", "The current validation results do not meet the publish conditions."))}</p>
      </section>
    `
  }
  return `
    <section class="detail-card tone-pending">
      <div class="headline-row">
        <div>
          <p class="kicker">${escapeHtml(l("结果", "Result"))}</p>
          <h4>${escapeHtml(l("这条候选还在流程中", "This candidate is still in progress"))}</h4>
        </div>
        ${badge(candidateStatusLabel(job.status), "pending")}
      </div>
      <p class="soft-copy">${escapeHtml(l("继续关注验证反馈数量、平均分和拒绝数是否达到阈值。", "Keep watching result counts, average score, and rejection count against the thresholds."))}</p>
    </section>
  `
}

function renderFinalPage() {
  const items = filteredFinalSkills()
  const detail = selectedFinalSkillDetail()

  dom.finalList.innerHTML = items.length
    ? items.map(renderFinalSkillCard).join("")
    : renderFinalEmptyState()

  dom.finalDetail.innerHTML = detail
    ? renderFinalSkillDetail(detail)
    : renderFinalDetailEmptyState()
}

function renderFinalSkillCard(skill) {
  const active = String(skill.skill_id || "") === String(state.selectedFinalSkillId || "")
  const sync = compareLocalAndRemote(skill)
  const candidateCount = candidateRecordCount(skill.name)
  return `
    <article class="record-card ${active ? "active" : ""}" data-select-final-skill="${escapeHtml(skill.skill_id)}">
      <div class="card-head">
        <div>
          <p class="card-kicker">${escapeHtml(categoryLabel(skill.category))}</p>
          <h4>${escapeHtml(skill.name || l("未命名技能", "Unnamed Skill"))}</h4>
        </div>
        ${badge(`v${number(skill.current_version || 0)}`, "published")}
      </div>
      <p class="card-copy">${escapeHtml(clip(skill.description || l("这个技能还没有描述。", "This skill does not have a description yet."), 110))}</p>
      <div class="chip-row">
        ${tag(sync.label)}
        ${tag(l("历史版本 {count}", "Versions {count}", { count: number(visibleVersionCount(skill)) }))}
        ${tag(l("候选记录 {count}", "Candidates {count}", { count: number(candidateCount) }))}
        ${skill.uploaded_at || skill.remote_updated_at ? tag(l("发布时间 {time}", "Published {time}", { time: formatStamp(skill.uploaded_at || skill.remote_updated_at) })) : ""}
      </div>
    </article>
  `
}

function renderFinalSkillDetail(skill) {
  const sync = compareLocalAndRemote(skill)
  return `
    <div class="stack">
      <section class="detail-card">
        <div class="detail-head">
          <div>
            <p class="kicker">${escapeHtml(l("最终池技能", "Published Skill"))}</p>
            <h3>${escapeHtml(skill.name || l("未命名技能", "Unnamed Skill"))}</h3>
            <p class="soft-copy">${escapeHtml(skill.description || l("这个技能还没有描述。", "This skill does not have a description yet."))}</p>
          </div>
          <div class="chip-row">
            ${badge(`v${number(skill.current_version || 0)}`, "published")}
            ${badge(sync.label, sync.tone)}
          </div>
        </div>
        <div class="mini-grid">
          ${renderMiniCard(l("当前正式版", "Current Official Version"), `v${number(skill.current_version || 0)}`, l("共享库当前正式版本。", "Current official version in the shared pool."))}
          ${renderMiniCard(l("历史版本", "Version Count"), number(visibleVersionCount(skill)), l("当前能看到的共享版本数量。", "Number of shared versions currently visible."))}
          ${renderMiniCard(l("最近发布时间", "Latest Publish Time"), formatStamp(skill.uploaded_at || skill.remote_updated_at || skill.updated_at), l("最新一次进入共享库的时间。", "When it most recently entered the shared pool."))}
          ${renderMiniCard(l("本地状态", "Local Status"), sync.label, l("这份共享正式版和你本地内容当前是什么关系。", "How this published version relates to your local content."))}
        </div>
      </section>
      ${renderVersionCompare(skill, {
        scope: "final",
        title: l("共享版本对比", "Shared Version Comparison"),
        copy: l("可以在这里对照 v1 / v2 / v3 等历史版本，也可以把本地当前内容拉进来比较。", "Compare v1 / v2 / v3 and other history versions here, and also compare them with your current local content."),
        includeLocal: true,
      })}
      ${renderVersionTimeline(skill, {
        title: l("共享版本时间线", "Shared Version Timeline"),
        copy: l("这个技能是如何一路发布到当前正式版的。", "How this skill evolved into the current official version."),
      })}
      ${renderCandidateLinks(skill.name)}
      ${renderRelatedSessions(skill.related_sessions, { emptyTitle: l("最近没有观测到相关会话", "No related sessions were observed recently") })}
    </div>
  `
}

function renderVersionCompare(skill, options) {
  const selection = compareState(skill, options.scope, { includeLocal: options.includeLocal })
  if (!selection.entries.length) {
    return `
      <section class="detail-card">
        <div class="headline-row">
          <div>
            <p class="kicker">${escapeHtml(options.title)}</p>
            <h4>${escapeHtml(l("当前没有可比较的版本", "No versions are available for comparison"))}</h4>
          </div>
        </div>
      </section>
    `
  }

  return `
    <section class="detail-card">
      <div class="headline-row">
        <div>
          <p class="kicker">${escapeHtml(options.title)}</p>
          <h4>${escapeHtml(options.copy)}</h4>
        </div>
        ${badge(l("{count} 个版本", "{count} versions", { count: number(selection.entries.length) }), "neutral")}
      </div>
      <div class="version-stack">
        <div class="selector-row">
          <label>
            <span class="field-label">${escapeHtml(l("左侧版本", "Left Version"))}</span>
            <select data-compare-field="primary" data-compare-scope="${escapeHtml(options.scope)}" data-skill-id="${escapeHtml(skill.skill_id)}">
              ${selection.entries.map((entry) => `
                <option value="${escapeHtml(entry.key)}" ${entry.key === selection.primary?.key ? "selected" : ""}>
                  ${escapeHtml(versionEntryLabel(entry))}
                </option>
              `).join("")}
            </select>
          </label>
          <label>
            <span class="field-label">${escapeHtml(l("右侧版本", "Right Version"))}</span>
            <select data-compare-field="compare" data-compare-scope="${escapeHtml(options.scope)}" data-skill-id="${escapeHtml(skill.skill_id)}">
              ${selection.entries.map((entry) => `
                <option value="${escapeHtml(entry.key)}" ${entry.key === selection.compare?.key ? "selected" : ""}>
                  ${escapeHtml(versionEntryLabel(entry))}
                </option>
              `).join("")}
            </select>
          </label>
        </div>
        <div class="version-pills">
          ${selection.entries.map((entry) => `
            <button
              class="version-pill ${entry.key === selection.primary?.key ? "active" : ""}"
              data-pick-version="${escapeHtml(entry.key)}"
              data-compare-scope="${escapeHtml(options.scope)}"
              data-compare-field="primary"
              data-skill-id="${escapeHtml(skill.skill_id)}"
              type="button"
            >
              ${escapeHtml(entry.label)}
            </button>
          `).join("")}
        </div>
        <div class="doc-grid">
          ${renderVersionDoc(selection.primary, l("左侧版本", "Left Version"))}
          ${renderVersionDoc(selection.compare, l("右侧版本", "Right Version"))}
        </div>
      </div>
    </section>
  `
}

function versionEntryLabel(entry) {
  const parts = [entry.label]
  if (entry.timestamp) {
    parts.push(formatStamp(entry.timestamp))
  }
  if (entry.action && entry.action !== "local") {
    parts.push(actionLabel(entry.action))
  }
  return parts.join(" · ")
}

function renderVersionDoc(entry, title) {
  if (!entry) {
    return `<div class="empty-state">${escapeHtml(l("没有可展示的版本。", "No version is available to display."))}</div>`
  }
  return `
    <article class="doc-card">
      <div class="doc-head">
        <div>
          <p class="kicker">${escapeHtml(title)}</p>
          <h4>${escapeHtml(entry.label)}</h4>
        </div>
        <div class="chip-row">
          ${badge(entry.source === "local" ? l("本地", "Local") : l("共享库", "Shared Pool"), entry.source === "local" ? "neutral" : "published")}
        </div>
      </div>
      <div class="panel-body">
        <div class="chip-row">
          ${entry.timestamp ? tag(formatStamp(entry.timestamp)) : ""}
          ${entry.action && entry.action !== "local" ? tag(actionLabel(entry.action)) : ""}
        </div>
      </div>
      ${String(entry.document || "").trim()
        ? `<pre>${escapeHtml(String(entry.document).trim())}</pre>`
        : `<div class="empty-state">${escapeHtml(l("这个版本只有元数据，没有留下文档快照。", "This version only has metadata and no saved document snapshot."))}</div>`}
    </article>
  `
}

function renderVersionTimeline(skill, options = {}) {
  const entries = buildVersionEntries(skill, { includeLocal: false }).filter((entry) => entry.source === "shared")
  if (!entries.length) {
    return `
      <section class="detail-card">
        <div class="headline-row">
          <div>
            <p class="kicker">${escapeHtml(options.title || l("版本时间线", "Version Timeline"))}</p>
            <h4>${escapeHtml(l("当前没有可用的共享版本历史", "No shared version history is available yet"))}</h4>
          </div>
        </div>
      </section>
    `
  }
  return `
    <section class="detail-card">
      <div class="headline-row">
        <div>
          <p class="kicker">${escapeHtml(options.title || l("版本时间线", "Version Timeline"))}</p>
          <h4>${escapeHtml(options.copy || l("版本是如何一路走到当前正式版的", "How versions progressed into the current official version"))}</h4>
        </div>
      </div>
      <div class="version-grid">
        ${entries.map((entry) => `
          <article class="version-card ${entry.current ? "current" : ""}">
            <div class="version-head">
              <div>
                <h4>${escapeHtml(entry.label)}</h4>
                <p class="soft-copy">${escapeHtml(entry.timestamp ? formatStamp(entry.timestamp) : l("无时间", "No time"))}</p>
              </div>
              ${badge(entry.current ? l("当前正式版", "Current") : l("历史版本", "History"), entry.current ? "published" : "neutral")}
            </div>
            <div class="chip-row">
              ${entry.action && entry.action !== "local" ? tag(actionLabel(entry.action)) : ""}
            </div>
          </article>
        `).join("")}
      </div>
    </section>
  `
}

function renderSharingNotConnectedNotice(copy) {
  return `
    <section class="detail-card tone-pending">
      <div class="headline-row">
        <div>
          <p class="kicker">${escapeHtml(l("共享状态", "Sharing"))}</p>
          <h4>${escapeHtml(l("当前没有接入共享库", "The shared pool is not connected"))}</h4>
        </div>
      </div>
      <p class="soft-copy">${escapeHtml(copy)}</p>
    </section>
  `
}

function renderCandidateEmptyState() {
  if (!sharingEnabled()) {
    return `
      <div class="empty-state">
        ${escapeHtml(l("当前没有启用共享库，所以候选池为空。接入共享库之后，这里才会显示候选、验证反馈和发布时间线。", "Sharing is not enabled, so the candidate pool is empty. After connecting the shared pool, candidates, validation feedback, and publish events will appear here."))}
      </div>
    `
  }
  return `<div class="empty-state">${escapeHtml(l("当前筛选条件下没有候选技能。", "No candidate skills match the current filter."))}</div>`
}

function renderCandidateDetailEmptyState() {
  if (!sharingEnabled()) {
    return `
      <div class="empty-state">
        ${escapeHtml(l("当前环境未启用共享与验证链路，所以没有候选详情可看。", "Sharing and validation are not enabled in this environment, so candidate details are unavailable."))}
      </div>
    `
  }
  return `<div class="empty-state">${escapeHtml(l("选择一个候选技能，查看它的验证轨迹和会话来源。", "Select a candidate skill to inspect validation progress and source sessions."))}</div>`
}

function renderFinalEmptyState() {
  if (!sharingEnabled()) {
    return `
      <div class="empty-state">
        ${escapeHtml(l("当前没有启用共享库，所以最终池为空。接入共享库后，这里会显示共享技能和版本历史。", "Sharing is not enabled, so the final pool is empty. After connecting the shared pool, published skills and version history will appear here."))}
      </div>
    `
  }
  return `<div class="empty-state">${escapeHtml(l("当前筛选条件下没有最终池技能。", "No published skills match the current filter."))}</div>`
}

function renderFinalDetailEmptyState() {
  if (!sharingEnabled()) {
    return `
      <div class="empty-state">
        ${escapeHtml(l("当前环境未启用共享库，所以还没有可查看的最终池详情。", "The shared pool is not enabled in this environment, so no published-skill details are available."))}
      </div>
    `
  }
  return `<div class="empty-state">${escapeHtml(l("选择一个最终池技能，查看版本历史、发布链路和相关会话。", "Select a published skill to inspect version history, release flow, and related sessions."))}</div>`
}

function renderSessionsPage() {
  const items = filteredSessions()
  const detail = selectedSessionDetail()

  dom.sessionList.innerHTML = items.length
    ? items.map(renderSessionCard).join("")
    : `<div class="empty-state">${escapeHtml(l("当前筛选条件下没有会话。", "No sessions match the current filter."))}</div>`

  dom.sessionDetail.innerHTML = detail
    ? renderSessionDetail(detail)
    : `<div class="empty-state">${escapeHtml(l("选择一个会话，查看逐轮内容、关联技能，以及它是否进入候选池。", "Select a session to inspect turn-by-turn content, related skills, and whether it entered the candidate pool."))}</div>`
}

function renderSessionCard(session) {
  const active = String(session.session_id || "") === String(state.selectedSessionId || "")
  return `
    <article class="record-card ${active ? "active" : ""}" data-select-session="${escapeHtml(session.session_id)}">
      <div class="card-head">
        <div>
          <p class="card-kicker">${escapeHtml(formatStamp(session.timestamp))}</p>
          <h4>${escapeHtml(session.session_id || l("未知会话", "Unknown Session"))}</h4>
        </div>
        ${badge(sourceLabel(session.source || "local"), String(session.source || "").toLowerCase() === "local" ? "neutral" : "published")}
      </div>
      <p class="card-copy">${escapeHtml(clip(session.prompt_preview || session.response_preview || l("没有摘要。", "No summary."), 120))}</p>
      <div class="chip-row">
        ${tag(outcomeLabel(session.outcome))}
        ${tag(l("回合 {count}", "Turns {count}", { count: number(session.num_turns || 0) }))}
        ${tag(l("技能 {count}", "Skills {count}", { count: number((session.skill_names || []).length) }))}
      </div>
    </article>
  `
}

function renderSessionDetail(session) {
  const links = Array.isArray(session.links) ? session.links : []
  const turns = Array.isArray(session.turns) ? session.turns : []
  const uniqueSkillNames = [...new Set([
    ...(Array.isArray(session.skill_names) ? session.skill_names : []),
    ...links.map((item) => item.skill_name),
  ].filter(Boolean))]
  const linkedJobs = jobsForSession(session.session_id)

  return `
    <div class="stack">
      <section class="detail-card">
        <div class="detail-head">
          <div>
            <p class="kicker">${escapeHtml(l("会话详情", "Session Detail"))}</p>
            <h3>${escapeHtml(session.session_id || l("未知会话", "Unknown Session"))}</h3>
            <p class="soft-copy">${escapeHtml(session.prompt_preview || session.response_preview || l("没有摘要。", "No summary."))}</p>
          </div>
          <div class="chip-row">
            ${badge(sourceLabel(session.source || "local"), String(session.source || "").toLowerCase() === "local" ? "neutral" : "published")}
            ${badge(outcomeLabel(session.outcome), "neutral")}
          </div>
        </div>
        <div class="mini-grid">
          ${renderMiniCard(l("时间", "Time"), formatStamp(session.timestamp), l("这条会话的时间戳", "Timestamp of this session"))}
          ${renderMiniCard(l("回合数", "Turns"), number(session.num_turns || 0), l("当前会话的轮次数量。", "Number of turns in this session."))}
          ${renderMiniCard(l("自动评分", "Auto Score"), formatScore(session.avg_prm_score), l("如果系统做了自动评分，会在这里显示。", "Displayed when automatic scoring is available."))}
          ${renderMiniCard(l("关联候选", "Related Candidates"), number(linkedJobs.length), linkedJobs.length ? l("这条会话被哪些候选技能引用过。", "Which candidate skills referenced this session.") : l("当前没有候选技能显式引用它。", "No candidate skill explicitly references this session."))}
        </div>
      </section>
      <section class="detail-card">
        <div class="headline-row">
          <div>
            <p class="kicker">${escapeHtml(l("关联技能", "Related Skills"))}</p>
            <h4>${escapeHtml(l("这条会话中出现过哪些技能", "Which skills appeared in this session"))}</h4>
          </div>
        </div>
        <div class="chip-row">
          ${uniqueSkillNames.length
            ? uniqueSkillNames.map(renderSessionSkillButton).join("")
            : `<span class="soft-copy">${escapeHtml(l("这个会话没有显式记录关联技能。", "This session has no explicitly recorded related skills."))}</span>`}
        </div>
      </section>
      <section class="detail-card">
        <div class="headline-row">
          <div>
            <p class="kicker">${escapeHtml(l("候选引用", "Candidate Links"))}</p>
            <h4>${escapeHtml(l("这个会话是否参与过候选技能的生成或验证", "Whether this session participated in candidate generation or validation"))}</h4>
          </div>
        </div>
        ${linkedJobs.length
          ? `
            <div class="stack">
              ${linkedJobs.map((job) => `
                <button class="jump-card" data-open-view="candidate" data-select-candidate="${escapeHtml(job.job_id)}" type="button">
                  <div class="headline-row">
                    <div>
                      <strong>${escapeHtml(job.skill_name || l("未命名技能", "Unnamed Skill"))}</strong>
                      <p class="soft-copy">${escapeHtml(job.job_id)} · ${escapeHtml(candidateStatusLabel(job.status))}</p>
                    </div>
                    ${badge(actionLabel(job.proposed_action), toneForStatus(job.status))}
                  </div>
                </button>
              `).join("")}
            </div>
          `
          : `<div class="empty-state">${escapeHtml(l("这条会话当前没有被候选池显式引用。", "This session is not explicitly referenced by the candidate pool."))}</div>`}
      </section>
      <section class="detail-card">
        <div class="headline-row">
          <div>
            <p class="kicker">${escapeHtml(l("逐轮内容", "Turn-by-Turn Content"))}</p>
            <h4>${escapeHtml(l("用户输入与模型回复片段", "User Prompts and Model Responses"))}</h4>
          </div>
          ${badge(l("{count} 轮", "{count} turns", { count: number(turns.length) }), "neutral")}
        </div>
        <div class="turn-stack">
          ${turns.length
            ? turns.map(renderTurnCard).join("")
            : `<div class="empty-state">${escapeHtml(l("这个会话没有可展示的 turn 记录。", "This session has no turn records to display."))}</div>`}
        </div>
      </section>
    </div>
  `
}

function renderSessionSkillButton(skillName) {
  const skill = findSkillByName(skillName)
  if (!skill) {
    return tag(skillName)
  }
  if (skill.has_local) {
    return `
      <button class="ghost" data-open-view="local" data-select-local-skill="${escapeHtml(skill.skill_id)}" type="button">
        ${escapeHtml(skillName)}
      </button>
    `
  }
  return `
    <button class="ghost" data-open-view="final" data-select-final-skill="${escapeHtml(skill.skill_id)}" type="button">
      ${escapeHtml(skillName)}
    </button>
  `
}

function renderTurnCard(turn) {
  const injected = normalizeSkillNames(turn.injected_skills)
  const read = normalizeSkillNames(turn.read_skills)
  const modified = normalizeSkillNames(turn.modified_skills)
  return `
    <article class="turn-card">
      <div class="turn-head">
        <div>
          <p class="kicker">${escapeHtml(l("第 {turn} 轮", "Turn {turn}", { turn: String(turn.turn_num || "-") }))}</p>
        </div>
        ${turn.prm_score != null ? badge(l("评分 {score}", "Score {score}", { score: formatScore(turn.prm_score) }), "neutral") : ""}
      </div>
      <div class="turn-block">
        <span class="turn-label">${escapeHtml(l("用户输入", "User Prompt"))}</span>
        <pre>${escapeHtml(String(turn.prompt_text || "").trim() || l("(空)", "(empty)"))}</pre>
      </div>
      <div class="turn-block">
        <span class="turn-label">${escapeHtml(l("模型回复", "Model Response"))}</span>
        <pre>${escapeHtml(String(turn.response_text || "").trim() || l("(空)", "(empty)"))}</pre>
      </div>
      <div class="chip-row">
        ${injected.map((item) => tag(l("注入 {name}", "Injected {name}", { name: item }))).join("")}
        ${read.map((item) => tag(l("读取 {name}", "Read {name}", { name: item }))).join("")}
        ${modified.map((item) => tag(l("修改 {name}", "Modified {name}", { name: item }))).join("")}
      </div>
    </article>
  `
}

async function runOperation(op) {
  if (state.loading) {
    return
  }
  const operationMap = {
    "toggle-locale": async () => {
      setLocale(state.locale === "en" ? "zh" : "en")
    },
    refresh: async () => {
      await refreshData({ notice: l("已刷新看板数据。", "Dashboard data refreshed.") })
    },
    "sync-projection": async () => {
      const response = await getJson("/api/v1/sync", { method: "POST" })
      const summary = response.summary || {}
      await refreshData({
        notice: l("已重建投影：{skills} 个技能，{sessions} 个会话。", "Snapshot rebuilt: {skills} skills, {sessions} sessions.", {
          skills: number(summary.skills || 0),
          sessions: number(summary.sessions || 0),
        }),
      })
    },
  }
  const handler = operationMap[op]
  if (!handler) {
    return
  }
  setLoading(true)
  try {
    await handler()
  } catch (error) {
    showMessage("error", error.message || l("操作失败", "Operation failed"))
  } finally {
    setLoading(false)
  }
}

async function selectLocalSkill(skillId, { openView = false } = {}) {
  state.selectedLocalSkillId = skillId || ""
  if (openView) {
    state.activeView = "local"
  }
  renderAll()
  await hydrateSelections({ local: true, final: false, session: false, candidate: false })
  renderAll()
}

async function selectFinalSkill(skillId, { openView = false } = {}) {
  state.selectedFinalSkillId = skillId || ""
  if (openView) {
    state.activeView = "final"
  }
  renderAll()
  await hydrateSelections({ local: false, final: true, session: false, candidate: false })
  renderAll()
}

async function selectCandidate(jobId, { openView = false } = {}) {
  state.selectedCandidateJobId = jobId || ""
  if (openView) {
    state.activeView = "candidate"
  }
  renderAll()
  await hydrateSelections({ local: false, final: false, session: false, candidate: true })
  renderAll()
}

async function selectSession(sessionId, { openView = false } = {}) {
  const sessionItem = findSessionById(sessionId)
  if (sessionItem && String(sessionItem.source || "").trim().toLowerCase() === "shared" && dom.sessionSource.value === "local") {
    dom.sessionSource.value = "all"
  }
  state.selectedSessionId = sessionId || ""
  if (openView) {
    state.activeView = "sessions"
  }
  ensureSelections()
  renderAll()
  await hydrateSelections({ local: false, final: false, session: true, candidate: false })
  renderAll()
}

async function onFilterChange(kind) {
  ensureSelections()
  renderAll()
  if (kind === "local" && state.selectedLocalSkillId) {
    await hydrateSelections({ local: true, final: false, session: false, candidate: false })
  } else if (kind === "candidate" && state.selectedCandidateJobId) {
    await hydrateSelections({ local: false, final: false, session: false, candidate: true })
  } else if (kind === "final" && state.selectedFinalSkillId) {
    await hydrateSelections({ local: false, final: true, session: false, candidate: false })
  } else if (kind === "session" && state.selectedSessionId) {
    await hydrateSelections({ local: false, final: false, session: true, candidate: false })
  }
  renderAll()
}

function setActiveView(view) {
  state.activeView = view
  renderAll()
}

async function handleDocumentClick(event) {
  const origin = event.target instanceof Element ? event.target : null
  if (!origin) {
    return
  }
  const target = origin.closest(
    "button, [data-select-local-skill], [data-select-final-skill], [data-select-candidate], [data-select-session], [data-view]"
  )
  if (!(target instanceof HTMLElement)) {
    return
  }

  if (target.dataset.view) {
    setActiveView(target.dataset.view)
    return
  }

  if (target.dataset.openView) {
    state.activeView = target.dataset.openView
  }

  if (target.dataset.op) {
    await runOperation(target.dataset.op)
    return
  }

  if (target.dataset.selectLocalSkill) {
    await selectLocalSkill(target.dataset.selectLocalSkill, { openView: true })
    return
  }

  if (target.dataset.selectFinalSkill) {
    await selectFinalSkill(target.dataset.selectFinalSkill, { openView: true })
    return
  }

  if (target.dataset.selectCandidate) {
    await selectCandidate(target.dataset.selectCandidate, { openView: true })
    return
  }

  if (target.dataset.selectSession) {
    await selectSession(target.dataset.selectSession, { openView: true })
    return
  }

  if (target.dataset.pickVersion) {
    setCompareSelection(
      target.dataset.compareScope,
      target.dataset.skillId,
      target.dataset.compareField || "primary",
      target.dataset.pickVersion
    )
    renderAll()
  }
}

function handleDocumentChange(event) {
  const target = event.target
  if (!(target instanceof HTMLSelectElement)) {
    return
  }
  if (target.dataset.compareField) {
    setCompareSelection(
      target.dataset.compareScope,
      target.dataset.skillId,
      target.dataset.compareField,
      target.value
    )
    renderAll()
  }
}

function bindEvents() {
  dom.localSearch.addEventListener("input", () => {
    void onFilterChange("local")
  })
  dom.candidateSearch.addEventListener("input", () => {
    void onFilterChange("candidate")
  })
  dom.candidateStatus.addEventListener("change", () => {
    void onFilterChange("candidate")
  })
  dom.finalSearch.addEventListener("input", () => {
    void onFilterChange("final")
  })
  dom.sessionSearch.addEventListener("input", () => {
    void onFilterChange("session")
  })
  dom.sessionSource.addEventListener("change", () => {
    void onFilterChange("session")
  })

  document.addEventListener("click", (event) => {
    void handleDocumentClick(event)
  })
  document.addEventListener("change", handleDocumentChange)
}

async function init() {
  bindEvents()
  await refreshData()
}

void init()
