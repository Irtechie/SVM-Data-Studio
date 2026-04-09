# AGENTS

## Default Work Rule

- Use the repo-local `todo.md` as the current execution queue.
- If the user gives a direct instruction, do that first.
- If the current task runs out of clear next steps, read `todo.md` before inventing new work.
- Do not auto-execute from `Parked / Cold Storage`.
- Do not treat `Human Required` items as agent-actionable until the human part is complete.

## Repo Files

- `todo.md`: approved active work
- `todo_rules.md`: runner conventions
- `todo_done.md`: completed history

## Fallback Behavior

- If `todo.md` exists, continue with the highest-value ready task.
- If `todo.md` does not exist, ask before creating a new task queue.
