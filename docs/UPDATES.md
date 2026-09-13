# EdgeSched 更新记录

本文件是 `legacy/mvp_core`（EdgeSched 独立仓库）的**详细**更新记录，与父仓库 `d:\AI\xiaoyou-core\UPDATES.md` 无关，两者不要互写。

## 记录分工

| 载体 | 写什么 | 不写什么 |
| --- | --- | --- |
| `docs/UPDATES.md`（本文件） | 背景、根因、方案、影响范围、验证方式 | — |
| git commit 主题行 | 一句话：**这次改动做了什么** | 文件清单（`git show --stat` 可查）、时间戳（`git log` 自带） |
| commit 正文（脚本 `-Detail`） | 只有需要随代码留存的过程信息时才写 | 大段复盘（写本文件） |

## 条目格式

```
## YYYY-MM-DD 标题

- 背景：为什么改
- 改动：改了哪些模块/口径，关键决策是什么
- 验证：跑了什么测试、得到什么结果（Mock 与真实 GPU/Forge 必须区分）
- 备注：遗留问题或后续动作
```

倒序追加，新条目放最上面。

## 2026-09-13 同步脚本自动提交必须写清"做了什么"

- 背景：`scripts/sync_to_wsl2_safe.ps1` 同步后固定提交 `auto: 同步后自动提交（时间戳）`，历史 commit 无法回答"这次改了什么、为什么改"；本仓库也没有 UPDATES.md 承载详细记录，追溯成本很高。
- 改动：
  - 脚本新增 `-Message`（必填）与 `-Detail`（可选）参数，提交主题行直接使用 `-Message`，`-Detail` 作为 commit 正文。
  - 未给 `-Message`、或 `-Message` 命中无信息黑名单（auto/update/更新/自动提交/同步 等）时直接 `exit 2`，且在 rsync 之前校验，避免白跑一次同步。
  - commit 改用 UTF-8 无 BOM 临时文件 + `git commit -F`，避免 Windows 控制台编码导致中文摘要乱码。
  - 提交前回显本次变更清单（最多 20 项），便于发现混入的无关文件。
  - 新增 `docs/UPDATES.md` 承载详细记录。
- 验证：`tests/scripts/verify_wsl_sync_policy.py` 增加断言，锁定 `-Message` 必填校验与黑名单存在，防止回退成无信息提交。
- 备注：工作区存在其他会话/其他 agent 的未提交改动时，仍应先用 `-SkipGit` 同步，再按需单独提交，避免把无关改动一起推上去。
