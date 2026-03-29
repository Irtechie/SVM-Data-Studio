# Todo Rules

These rules apply to domain runner files across the repo.

Use them when maintaining a single live task runner for an active effort.

## Purpose
- Keep one current runner file per active workstream.
- Reuse that file throughout the day instead of spawning many temporary ledgers.
- Make progress visible without needing chat context to know what is done, active, or blocked.

## Control Files
Use these files as distinct layers, not as interchangeable notes:

- `program.md`
  - defines the improvement method
  - records experiment rules, evaluation logic, mutation boundaries, and promotion criteria
  - should govern how improvement proposals are generated and judged
  - should not be treated as the live execution queue
- `todo.md`
  - the single live execution queue for approved work
  - only actionable, currently approved tasks belong here
- `todo_done.md`
  - the historical completion ledger
  - move completed or superseded work here once the live runner gets noisy
- `Parked / Cold Storage`
  - a section inside `todo.md` for discovered work that should not execute yet
  - use it for autoresearch findings, follow-on ideas, future branches, and deferred work
  - nothing in this section should run until a human promotes it into `Active Tasks`

Promotion rule:
- discovered work may be written into `Parked / Cold Storage`
- discovered work must not be auto-executed from there
- only human-reviewed items should move from cold storage into the active queue

## Status Markers
- Active task: `- [ ]`
- Completed task: `- [x] ~~task text~~`
- Blocked task: `- [!]`
- Optional or future task: `- [-]`

## Optional Task Metadata
Use short metadata lines directly under a task when dependency or readiness state matters.

Supported fields:
- `Task ID: <stable-id>`
- `Ready: yes|no`
- `Depends on: <id>, <id>`
- `Discovered from: <id or short note>`
- `Validation: <short proof>`

Rules:
- Omit metadata when the defaults are obvious.
- Default assumptions are:
  - active tasks are `Ready: yes`
  - completed tasks are not actionable
  - blocked tasks are `Ready: no`
  - missing `Depends on` means no declared dependency
- Keep `Task ID` stable once introduced so later notes and follow-on tasks can refer to it.
- Use `Discovered from` when a task was uncovered while executing another task.

## Update Rules
- Keep completed tasks in place and strike them through.
- Do not delete completed tasks from the runner.
- Append short validation results under the task.
- If a task is fully done, mark it complete in the runner instead of creating a separate marker file.
- If a task splits into smaller tasks, add them directly below the parent task.
- Keep blockers explicit in a `Blocked` section.
- Keep one `Work Log` section with short dated entries.
- Prefer marking a task blocked with explicit `Depends on` instead of silently letting it drift.

Long-lived exception:
- If a runner becomes too slow to scan, move old completed/history content into a sibling `todo_done.md`.
- Leave a short pointer in the live runner instead of duplicating the full history.
- The active runner should stay optimized for current actionable work.
- If work is merely discovered or proposed, move it to `Parked / Cold Storage` instead of the done log.
- If `program.md` or an offline improvement lane surfaces new ideas, record them in cold storage first unless a human explicitly promotes them.

## Runner Structure
Recommended sections:
- `Purpose`
- `Objective`
- `Current Focus`
- `Current Truth`
- `Success Criteria`
- `Active Tasks`
- `Parked / Cold Storage`
- `Blocked`
- `Work Log`

## Execution Expectations
- Work top to bottom unless a blocker forces reordering.
- Validate after each meaningful task.
- Prefer concrete "done when" conditions over vague prose.
- Use the runner as the current source of truth, not chat history.
- If dependency metadata exists, prefer the highest-value `Ready: yes` tasks before blocked ones.
- Do not execute from `Parked / Cold Storage`.
- If `program.md` produces an improvement candidate, capture it in cold storage and wait for human promotion unless the runner already explicitly authorizes it.

## Ready-Task Query
When a runner uses the metadata above, the ready queue can be queried with:

```bash
python3 scripts/task_runner_ready.py codex/godot/todo.md
```

Use `--json` when another tool or script needs machine-readable output.

## Naming
Suggested names:
- `todo.md`
- `TASK_RUNNER_<workstream>.md`

Prefer `todo.md` when one runner is clearly the active focus for the folder.
Keep legacy names only where a folder already depends on them.

## Discovery
- Start from the repo-root [AGENTS.md](/home/admin/apps/Alpha/AGENTS.md) to route into the right domain.
- Once routed into a domain, look for a local `todo.md` first.
- If the domain still uses a legacy runner filename, use that local runner instead of inventing a new one.
- Use this top-level `todo_rules.md` as the shared convention file for all domain runners.
