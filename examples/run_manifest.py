"""The run manifest: what produced a run, written as one document and read back.

Asserts what `docs/run-manifest.md` claims:

* both halves are written — `manifest.yaml` to read and `manifest.json` to parse — and rewritten
  after EVERY mutation, so a run killed mid-loop still leaves a complete record;
* a dataset and a model are recorded by reference, not by copying their bytes;
* `capture_packages` writes the installed-package freeze as a SIDE artifact, which travels with
  the manifest through `artifacts()`;
* a path under `$DATA_ROOT` is recorded as `$DATA_ROOT/...`, so the manifest still points at the
  right thing on a machine that keeps its data somewhere else;
* `finish()` stamps the terminal status.

No tracker and no training framework: the manifest is a document, and this writes one by hand.
`log_run_context([tracker], ...)` is what fills and projects it in a real run — see
`examples/standalone_tracking.py`.

Run:  python examples/run_manifest.py
"""

import json
import os
import tempfile
from pathlib import Path

import yaml

from matrainer.tracking.manifest import RunManifest


def main() -> None:
    root = Path(tempfile.mkdtemp(prefix="matrainer_manifest_"))
    # The machine's data root. A real run reads this from the environment; setting it here is what
    # lets the example show the contraction rather than describe it.
    data_root = root / "store"
    (data_root / "imagefolder").mkdir(parents=True)
    os.environ["DATA_ROOT"] = str(data_root)

    # `handle_interrupts=False` keeps the example from installing signal handlers.
    manifest = RunManifest(run_dir=root / "run", experiment_name="manifest-demo", task="fit", handle_interrupts=False)

    manifest.set("params", {"lr": 1e-3, "batch_size": 32})
    manifest.record_dataset(
        kind="file",
        name=str(data_root / "imagefolder"),
        uri=f"file://{data_root / 'imagefolder'}",
        split="train",
        count=1200,
    )
    manifest.record_model(source="file", name="resnet18", uri="file:///models/resnet18.pth", role="checkpoint")
    manifest.capture_code()  # the git state of every editable distribution on sys.path
    manifest.capture_packages()  # every installed distribution and its version

    # Nothing above asked for a write: the manifest rewrites both files after every mutation.
    assert manifest.path.name == "manifest.yaml" and manifest.path.exists()
    assert manifest.json_path.name == "manifest.json" and manifest.json_path.exists()

    document = yaml.safe_load(manifest.path.read_text())
    assert document == json.loads(manifest.json_path.read_text()), "the two halves must agree"

    # The recorded dataset path is contracted against the root, in BOTH the name and the URI.
    dataset = manifest.section("datasets")[0]
    assert dataset["name"] == "$DATA_ROOT/imagefolder", dataset["name"]
    assert dataset["uri"] == "file://$DATA_ROOT/imagefolder", dataset["uri"]
    assert dataset["count"] == 1200

    # The freeze is a side artifact: a path the manifest carries, not a blob inside it.
    artifacts = manifest.artifacts()
    assert any(p.name == "requirements.txt" for p in artifacts), artifacts
    assert all(p.exists() for p in artifacts)
    # Every tracker receives these beside the manifest, so a remote-only backend is not left
    # holding references to files that exist on exactly one machine.

    manifest.finish("success")
    assert manifest.status == "success"
    written = yaml.safe_load(manifest.path.read_text())
    assert written["status"] == "success"  # top level: the first thing a reader looks for
    assert written["run"]["stopped_at"] is not None  # and when it ended

    print(f"manifest written to {manifest.directory}")
    for path in sorted(p.relative_to(manifest.directory) for p in manifest.directory.rglob("*") if p.is_file()):
        print(f"  {path}")
    print(f"sections recorded: {', '.join(k for k in document if document[k])}")


if __name__ == "__main__":
    main()
