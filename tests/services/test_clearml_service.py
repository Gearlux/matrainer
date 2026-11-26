"""The ClearML server as a workspace service (`matrainer/services/clearml_service.py`).

The server is upstream's docker-compose stack of seven containers, fetched at a pinned release and checked against a
pinned SHA-256. Upstream writes every host bind mount under `/opt/clearml`; the service's state must live in its own
data folder, so an override GENERATED from upstream's file moves each one there — a version bump cannot leave a mount
behind. Docker runs it, not the service manager's supervisor: the class has its own start, stop, status, info and
reset.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, List
from unittest.mock import MagicMock, patch

import pytest
import yaml

from matrainer.services import clearml_service as cs
from matrainer.services.clearml_service import CLEARML_SERVER_VERSION, COMPOSE_SHA256, UPSTREAM_HOST_ROOT, ClearML
from tests.services.context import Context, Failed

UPSTREAM = """\
services:
  apiserver:
    image: allegroai/clearml:latest
    volumes:
      - /opt/clearml/logs:/var/log/clearml
      - /opt/clearml/config:/opt/clearml/config
    ports:
      - "8008:8008"
  redis:
    image: redis:6.2
    volumes:
      - /opt/clearml/data/redis:/data
  webserver:
    image: allegroai/clearml:latest
    ports:
      - "8080:80"
"""


@pytest.fixture
def files(tmp_path: Path) -> cs.Files:
    """The service's files under a data folder holding upstream's compose already."""
    found = cs.Files(tmp_path / "clearml")
    found.data_dir.mkdir(parents=True)
    found.upstream.write_text(UPSTREAM)
    return found


def _override(files: cs.Files) -> Any:
    cs.write_override(files)
    return yaml.safe_load(files.override.read_text())


# --- the override -------------------------------------------------------------------------------------------------


def test_every_upstream_host_path_is_relocated_into_the_state_folder(files: cs.Files) -> None:
    mounts = [entry for spec in _override(files)["services"].values() for entry in spec["volumes"]]
    assert mounts, "the override produced no remaps at all"
    for entry in mounts:
        host = entry.split(":", 1)[0]
        assert not host.startswith(UPSTREAM_HOST_ROOT), entry
        assert str(files.state) in host, entry


def test_the_container_targets_are_preserved(files: cs.Files) -> None:
    """Compose merges volumes BY TARGET — a changed target would add a mount, not replace one."""
    document = _override(files)
    targets = {entry.split(":", 1)[1] for spec in document["services"].values() for entry in spec["volumes"]}
    assert targets == {"/var/log/clearml", "/opt/clearml/config", "/data"}


def test_services_without_bind_mounts_are_omitted(files: cs.Files) -> None:
    assert set(_override(files)["services"]) == {"apiserver", "redis"}


def test_the_override_is_derived_not_hardcoded(files: cs.Files) -> None:
    """A mount upstream adds later is picked up without editing this project."""
    files.upstream.write_text(
        UPSTREAM + "  newthing:\n    image: x\n    volumes:\n      - /opt/clearml/data/newthing:/mnt/new\n"
    )
    document = _override(files)
    assert str(files.state / "data" / "newthing") in document["services"]["newthing"]["volumes"][0]


def test_the_state_folders_are_created(files: cs.Files) -> None:
    """Docker would otherwise create them as root-owned, or fail on a read-only parent."""
    cs.write_override(files)
    assert (files.state / "logs").is_dir()
    assert (files.state / "data" / "redis").is_dir()


def test_the_override_carries_generator_provenance(files: cs.Files) -> None:
    header = cs.write_override(files).read_text()
    assert "AUTO-GENERATED" in header
    assert "matrainer/matrainer/services/clearml_service.py" in header
    assert "aisland clearml reset && aisland clearml start" in header


def test_no_fixed_user_goes_through_environment_variables(files: cs.Files) -> None:
    for spec in _override(files)["services"].values():
        assert "environment" not in spec


# --- the pinned download ------------------------------------------------------------------------------------------


def _response(body: bytes) -> MagicMock:
    response = MagicMock()
    response.read.return_value = body
    response.__enter__ = lambda s: s
    response.__exit__ = lambda *a: None
    return response


def test_a_checksum_mismatch_stops_and_writes_nothing(files: cs.Files, tmp_path: Path) -> None:
    files.upstream.unlink()
    ctx = Context(files.data_dir)
    with patch("urllib.request.urlopen", return_value=_response(b"services: {}\n")):
        with pytest.raises(Failed, match="Checksum mismatch"):
            cs.fetch_upstream(ctx, files)
    assert not files.upstream.exists()


def test_a_cached_file_is_not_fetched_again(files: cs.Files) -> None:
    with patch("urllib.request.urlopen", side_effect=AssertionError("must not download")):
        assert cs.fetch_upstream(Context(files.data_dir), files) == files.upstream


def test_the_pin_is_a_release_tag() -> None:
    """A digest pinned to a moving ref breaks on upstream's next commit."""
    assert CLEARML_SERVER_VERSION.startswith("v") and CLEARML_SERVER_VERSION not in ("master", "main")
    assert len(COMPOSE_SHA256) == 64


# --- what start checks first --------------------------------------------------------------------------------------


def test_without_clearml_start_names_the_command_that_adds_it(files: cs.Files) -> None:
    ctx = Context(files.data_dir, chosen=["mlflow"])
    with patch.object(cs, "_installed", lambda: False):
        with pytest.raises(Failed) as failed:
            ClearML().start(ctx)
    assert "      aisland workspace configure tracker set mlflow clearml\n" in str(failed.value)
    assert str(failed.value).startswith("ClearML is not installed in this environment.")


def test_without_the_docker_cli_start_says_how_to_install_it(files: cs.Files) -> None:
    with patch.object(cs, "_installed", lambda: True), patch("shutil.which", return_value=None):
        with pytest.raises(Failed, match="brew install docker"):
            ClearML().start(Context(files.data_dir))


def test_with_docker_stopped_start_names_the_command_that_starts_it(files: cs.Files) -> None:
    with patch.object(cs, "_installed", lambda: True), patch("shutil.which", return_value="/usr/bin/docker"):
        with patch("subprocess.run", return_value=MagicMock(returncode=1)):
            with pytest.raises(Failed, match="aisland docker start"):
                ClearML().start(Context(files.data_dir))


# --- the workspace .env -------------------------------------------------------------------------------------------


def test_the_hosts_are_published(files: cs.Files) -> None:
    ctx = Context(files.data_dir, env={"DATA_ROOT": "/data", "CLEARML_WEB_HOST": "http://localhost:1111"})
    cs.publish_hosts(ctx, 9999)
    assert ctx.env["CLEARML_WEB_HOST"] == "http://localhost:9999"
    assert ctx.env["CLEARML_API_HOST"].startswith("http://localhost:")
    assert ctx.env["DATA_ROOT"] == "/data"


def test_an_existing_login_is_reused(files: cs.Files) -> None:
    """A fresh password each start would not match the server that already has the old one."""
    ctx = Context(files.data_dir, env={"CLEARML_USERNAME": "me", "CLEARML_PASSWORD": "keepme"})
    assert cs.login(ctx) == {"CLEARML_USERNAME": "me", "CLEARML_PASSWORD": "keepme"}


def test_a_missing_login_is_generated_from_the_workspace_name_and_recorded(files: cs.Files) -> None:
    ctx = Context(files.data_dir, workspace_name="acme")
    made = cs.login(ctx)
    assert made["CLEARML_USERNAME"] == "acme"
    assert len(made["CLEARML_PASSWORD"]) >= 20  # generated, not a default anyone could guess
    assert ctx.env["CLEARML_PASSWORD"] == made["CLEARML_PASSWORD"]


# --- the fixed user's config file ---------------------------------------------------------------------------------


def test_the_fixed_user_goes_in_a_conf_file_where_the_container_reads_it(files: cs.Files) -> None:
    """Measured on v2.4.0: the environment-variable override cannot express a list of objects."""
    written = cs.write_apiserver_conf(files, {"CLEARML_USERNAME": "u", "CLEARML_PASSWORD": "p"})
    assert written == files.state / "config" / "apiserver.conf"
    body = written.read_text()
    assert "fixed_users" in body and "enabled: true" in body
    assert 'username: "u"' in body and 'password: "p"' in body


def test_the_conf_is_not_world_readable(files: cs.Files) -> None:
    written = cs.write_apiserver_conf(files, {"CLEARML_USERNAME": "u", "CLEARML_PASSWORD": "p"})
    assert written.stat().st_mode & 0o077 == 0


# --- credentials --------------------------------------------------------------------------------------------------


def test_minting_credentials_never_raises(files: cs.Files) -> None:
    """A tracking convenience must not fail a stack that is otherwise up."""
    with patch("urllib.request.urlopen", side_effect=OSError("connection refused")):
        assert cs.mint_credentials(Context(files.data_dir), {"CLEARML_USERNAME": "u", "CLEARML_PASSWORD": "p"}) is None


def test_minted_credentials_are_recorded(files: cs.Files) -> None:
    replies = [{"data": {"token": "tok"}}, {"data": {"credentials": {"access_key": "AK", "secret_key": "SK"}}}]
    ctx = Context(files.data_dir)
    with patch("urllib.request.urlopen", side_effect=lambda *a, **k: _response(json.dumps(replies.pop(0)).encode())):
        minted = cs.mint_credentials(ctx, {"CLEARML_USERNAME": "u", "CLEARML_PASSWORD": "p"})
    assert minted == {"CLEARML_API_ACCESS_KEY": "AK", "CLEARML_API_SECRET_KEY": "SK"}
    assert ctx.env["CLEARML_API_ACCESS_KEY"] == "AK"


# --- its own lifecycle --------------------------------------------------------------------------------------------


def test_the_compose_command_layers_upstream_then_the_override(files: cs.Files) -> None:
    argv = cs.compose(files, "up", "-d")
    assert argv[:3] == ["docker", "compose", "-f"]
    assert argv.index(str(files.upstream)) < argv.index(str(files.override))


def test_it_is_a_service_docker_runs(files: cs.Files) -> None:
    assert ClearML.kind == "service"
    assert ClearML.supervised is False


def test_stop_status_and_reset_before_any_start_touch_nothing(files: cs.Files) -> None:
    calls: List[Any] = []
    ctx = Context(files.data_dir)
    with patch("subprocess.run", side_effect=lambda *a, **k: calls.append(a)):
        ClearML().stop(ctx)
        ClearML().status(ctx)
    assert calls == []
    assert ctx.lines == [
        "ClearML has never been started here; nothing to stop.",
        "  ClearML         : Not configured (run `aisland clearml start`)",
    ]


def test_reset_takes_the_stack_down_and_deletes_its_state(files: cs.Files) -> None:
    cs.write_override(files)
    calls: List[Any] = []
    with patch("subprocess.run", side_effect=lambda argv, **k: calls.append(argv)):
        ClearML().reset(Context(files.data_dir))
    assert calls == [cs.compose(files, "down", "-v")]
    assert not files.upstream.exists() and not files.override.exists() and not files.state.exists()


def test_info_says_docker_runs_it_not_the_supervisor(files: cs.Files) -> None:
    ctx = Context(files.data_dir)
    ClearML().info(ctx)
    assert "  Lifecycle     : docker compose (NOT the process-compose supervisor)" in ctx.lines
    assert f"  Upstream      : clearml-server {CLEARML_SERVER_VERSION} (sha256 pinned)" in ctx.lines


def test_the_module_imports_nothing_of_the_service_manager() -> None:
    code = "import sys; sys.modules['aisland'] = None\nfrom matrainer.services.clearml_service import ClearML\n"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=Path(__file__).parent)
    assert result.returncode == 0, result.stderr


def test_both_services_are_declared_for_the_service_manager() -> None:
    from importlib.metadata import entry_points

    declared = {ep.name: ep.value for ep in entry_points(group="aisland.plugins")}
    assert declared["aim"] == "matrainer.services.aim_service:Aim"
    assert declared["clearml"] == "matrainer.services.clearml_service:ClearML"
