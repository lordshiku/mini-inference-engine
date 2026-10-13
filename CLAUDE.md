# Rules for this repo

- Read SPEC.md first. Work on exactly one milestone at a time, in order.
- Run `pytest` after every change. Do not move to the next milestone until the current acceptance test passes.
- Keep `transformers` out of `engine/` — it is only allowed in tests and benchmarks as the reference.
- Prefer simple, readable code over clever code. Every function in `engine/` gets a short docstring saying what it does and why.
- Commit after each passing milestone with message `M<n>: <what>`. Push.
- Append all benchmark results to BENCHMARKS.md with the GPU name and date.
- If stuck on one bug for more than 1 hour, write what you tried to NOTES.md and move to the fallback in SPEC.md if one exists; otherwise stop and leave a clear message.
- Never download other models or change the model. Never install packages without adding them to requirements.txt.
