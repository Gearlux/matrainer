import subprocess


def get_git_commit_hash() -> str | None:
    """Return the current git commit hash, appended with '-dirty' if there are uncommitted changes."""
    try:
        git_hash = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode("ascii")
            .strip()
        )
        # Check if there are uncommitted changes
        is_dirty = (
            subprocess.check_output(
                ["git", "status", "--short"], stderr=subprocess.DEVNULL
            )
            .decode("ascii")
            .strip()
            != ""
        )
        return git_hash + ("-dirty" if is_dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
