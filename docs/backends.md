# Choosing a tracking backend

What each backend costs to run, and the two that this package can start for you as services.

## Choosing a backend

The `Tracker` surface is the same everywhere, so the choice is about **where the results live** and
**what it costs to run**, not about the API. Matrainer's own selection criterion for what it
implements in-tree is *free and self-hostable* — you should be able to `pip install`, start a server
on your own machine, and see your runs, with no account and no licence key.

**Free, self-hosted, no account:**

| Backend | Library | Licence | Local UI | Status |
|---|---|---|---|---|
| `CsvTracker` | stdlib only | — | none (plain files) | ✅ shipped, the default |
| `MlflowTracker` | `mlflow` | Apache-2.0 | `mlflow server` | ✅ shipped |
| `TensorBoardTracker` | `torch.utils.tensorboard` | Apache-2.0 | `tensorboard --logdir` | ✅ shipped |
| `AimTracker` | `aim` | Apache-2.0 | `aim up` — no server infrastructure | ✅ shipped |
| `ClearMLTracker` | `clearml` | Apache-2.0 (self-hosted server via docker-compose) | full web UI — or `offline=True` for no server at all | ✅ shipped |

Other free and self-hostable options, **not** implemented here — the `Tracker` ABC makes any of them
a single new file if you need one (see [architecture record 4](architecture.md)):
`dvclive` (git-native experiment versioning), `trackio` (a Weights & Biases-API-compatible local
dashboard), `sacred` + Omniboard (needs a MongoDB), and `determined` (Apache-2.0, but it is a cluster
scheduler — heavy if all you want is tracking).

**Hosted services with a free tier** — usable, but the server is not something you can run yourself
for free, so a fresh checkout is not steered toward them:

| Service | Status | Note |
|---|---|---|
| Weights & Biases | ✅ `WandbTracker` shipped | free personal tier; self-hosting is the enterprise offering. `wandb offline` writes local run directories, but reading them back needs a server. |
| Neptune | not implemented | free individual tier, hosted; self-host is enterprise. |
| Comet | not implemented | free individual/academic tier, hosted; self-host is enterprise. Their Apache-2.0 *Opik* is LLM evaluation, not experiment tracking. |

Check current terms before relying on any hosted tier — they change more often than this file does.

**Rule of thumb.** Start on `CsvTracker` (works with zero extras). Add `MlflowTracker` when you want
a model registry and a queryable run history; add `AimTracker` when you mainly want to *compare*
many runs quickly and do not want to operate a server; add `ClearMLTracker` when you want dataset
and orchestration surfaces in the same place. `TensorBoardTracker` complements rather than replaces
any of them — it is a visualizer, not a metadata store. `TrackerGroup` means you never have to pick
just one.

## The tracking servers as workspace services (`matrainer.services`)

In an aisland workspace matrainer brings two services, found through its `aisland.plugins` entry points:

```bash
aisland aim start -b                 # the Aim UI on http://localhost:43800, over the service's own repository
aisland aim start -b --repo ./runs/aim   # another repository
aisland clearml start                # the ClearML server (Docker), on http://localhost:8080
aisland clearml info                 # where its compose files, state and login are
```

Neither installs its backend: without `aim` or `clearml` the start names the command that adds it to the `tracker`
selection, keeping the trackers already chosen. The Aim service writes the repository it serves into the workspace
`.env` as `AIM_REPO`, so a run's `AimTracker(save_dir="$AIM_REPO")` writes where the UI reads. The ClearML service
runs upstream's compose stack at a pinned release (its SHA-256 checked), keeps every container's data in the
service's folder, and writes the hosts, the login and freshly minted SDK credentials into `.env` — so a
`ClearMLTracker` connects with no further setup.
