# marainer — backlog

Open work for this project. Cross-cutting / multi-project initiatives live in the
workspace root `TASKS.md`. Completed items are not archived here — git history is the record.

- [ ] Implement Base Trainer class (PyTorch/Lightning). @high
- [ ] Implement Base Evaluator and Metric calculation. @high
- [ ] RecordStream Lineage tracking (manifest artifact logging). @medium
- [ ] **RecordStream FlowGraph — pass more than a Record (data-plane generalization)** @medium @feature — generalize the linear single-`Record` `Stream` to a flat `steps:` list of `{op, in, out}` over a `Context` of NAMED cells (a cell holds a `Record` OR a raw value — tensor/scalar/list/second-record), so an op/wire can carry more than one Record and ops gain N named in/out. Linear FlowGraph defaults to today's `Stream` byte-identically; metadata stays inside record-cells. Dissolves the `export.py` auto-Stash/Unstash fan-out hack, the ComposeRecord fan-in boundary, ConfigureOp/CaptureOutputOp metadata-spilling, and gives conditional data-routing a home. The runnable/workflow plane (above) is the sibling at the orchestration level. Decided 2026-06-17; deferred behind the workflow combinators.
