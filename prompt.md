@plan.md @activity.md @proj.md

You are implementing a research project on Typed Context via Persistent Rotation for prompt injection defense.

First read activity.md to see what was recently accomplished and what state the project is in.

Then open plan.md and find the single highest priority task where passes is false.

Before implementing, read the relevant sections of proj.md — it contains detailed specifications, code templates, expected results, and visualization specs. Use its code as a starting point but adapt as needed for the actual model/library versions.

Work on exactly ONE task:

1. Implement all steps listed for that task.
2. Write clean, minimal code. Prefer single-file scripts. No unnecessary abstractions.
3. If a step produces a figure, save it to outputs/ with the exact filename from plan.md.
4. Run the verification step at the end of the task. If it fails, debug and fix before moving on.
5. If GPU is unavailable or insufficient, implement the code to be correct and verify on a small synthetic input (2 samples, 2 layers). Add a comment noting it needs full GPU run.

After completing the task:

1. Append a dated progress entry to activity.md describing: what you implemented, which files were created/modified, what the verification result was, and any issues encountered.
2. Update that task's passes in plan.md from false to true.
3. Make one git commit for that task only with a clear message like "Phase 1: extract hidden states for probing".
4. Do not git init, do not change remotes, do not push.

ONLY WORK ON A SINGLE TASK PER ITERATION.

When ALL tasks have passes true, output <promise>COMPLETE</promise>
