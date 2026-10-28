"""A model package: the inference graph, its weights and its class names, as one folder.

Asserts what `docs/model-packages.md` claims:

* `PackageModel` writes a FOLDER — `package.yaml`, the graph files under `graphs/`, the weights
  under `weights/`, and `class_names.json`;
* the author's graph files are copied in their own layout, includes and all, never rewritten;
* `copy` holds the bytes (an edit of the source afterwards changes nothing) while `reference`
  records the path and follows every edit — the two settings are independent;
* `load_package_graph` puts each weights file in the slot `package.yaml` maps it to, so a
  consumer runs the package without knowing how the graph is wired.

The "model" here is a chain of recordstream `PutField` ops and the "weights" a text file, so the
example needs no torch and no training run. A real package's predictor is a model op and its
weights a checkpoint; nothing else differs.

Run:  python examples/model_package.py
"""

import tempfile
from pathlib import Path

from matrainer.model_package import PackageModel, load_package_graph, read_package

# The inference graph, as a project would write it: preprocessing, the predictor, postprocessing.
# `pre` lives in its own file to show that an include is carried into the package as a file.
PREPROCESS = """\
scale:
  op: !class:recordstream.ops.structure.PutField {key: scaled, value: 1}
"""

GRAPH = """\
pre:
  include: ../graphs/preprocess.steps.yaml
flow:
  pre:
    op: !class:recordstream.flow.subgraph.Subgraph
      steps: !ref:pre
  predict:
    op: !class:recordstream.ops.structure.PutField {key: weights, value: ""}
    from: pre
  post:
    op: !class:recordstream.ops.structure.PutField {key: label, value: bird}
    from: predict
outputs: post
"""

#: The dotted key of the predictor's weights slot, as the packaging config names it.
WEIGHTS_SLOT = "flow.predict.op.value"


def a_project_to_package(root: Path) -> Path:
    (root / "graphs").mkdir(parents=True)
    (root / "cfg").mkdir()
    (root / "runs").mkdir()
    (root / "graphs" / "preprocess.steps.yaml").write_text(PREPROCESS)
    (root / "cfg" / "model.yaml").write_text(GRAPH)
    (root / "runs" / "best.pth").write_text("weights v1")
    return root


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="matrainer_package_"))
    source = a_project_to_package(tmp / "project")

    package = Path(
        PackageModel(
            graph=str(source / "cfg" / "model.yaml"),
            weights={WEIGHTS_SLOT: str(source / "runs" / "best.pth")},
            class_names=["bird", "cat", "dog"],
            description="the toy detector of run a",
            output_dir=str(tmp / "package"),
        ).run()
    )

    # A folder, with the four things a consumer needs.
    assert (package / "package.yaml").is_file()
    assert (package / "class_names.json").is_file()
    # The author's layout is kept: the entry graph AND the file it includes, as they were written.
    assert (package / "graphs" / "cfg" / "model.yaml").is_file()
    assert (package / "graphs" / "graphs" / "preprocess.steps.yaml").is_file()
    assert (package / "weights" / "best.pth").read_text() == "weights v1"

    # `package.yaml` is plain data: every graph and weights file with its digest, the class names.
    document = read_package(package)
    assert document["class_names"] == ["bird", "cat", "dog"]
    assert document["description"] == "the toy detector of run a"
    assert all("sha256" in entry for entry in document["weights"].values()), document["weights"]

    # The weights land in the slot, so the caller loads the graph without knowing the wiring —
    # the LIVE predictor op is holding the file by the time the graph is handed back.
    loaded = load_package_graph(str(package))
    assert loaded.document["flow"]["predict"]["op"].value.endswith("best.pth")
    # `request` is the document `confluid.load` was given, for a reader that needs the same
    # inputs at another stage (torpaido's compiler is one).
    assert loaded.request[WEIGHTS_SLOT].endswith("best.pth")

    # `copy` took the bytes: editing the source afterwards cannot change this package.
    (source / "runs" / "best.pth").write_text("weights v2")
    assert (package / "weights" / "best.pth").read_text() == "weights v1"

    # `reference` records the path instead, and follows the edit — the opposite trade.
    referenced = Path(
        PackageModel(
            graph=str(source / "cfg" / "model.yaml"),
            weights={WEIGHTS_SLOT: str(source / "runs" / "best.pth")},
            weight_files="reference",
            output_dir=str(tmp / "referenced"),
        ).run()
    )
    assert not (referenced / "weights").exists(), "a referenced package holds no weights"
    slot = load_package_graph(str(referenced)).document["flow"]["predict"]["op"].value
    assert Path(slot).read_text() == "weights v2", "it follows the source"

    print(f"package written to {package}")
    for path in sorted(p.relative_to(package) for p in package.rglob("*") if p.is_file()):
        print(f"  {path}")
    print("copy holds the weights it was given; reference follows the file it names")


if __name__ == "__main__":
    main()
