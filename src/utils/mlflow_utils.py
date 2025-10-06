from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional
import os
import subprocess
from datetime import datetime
import mlflow
from loguru import logger

from .general import get_project_root


def setup_mlflow(experiment: str = "mma_default", tracking_dir: Optional[str] = None) -> None:
    """
    Ensure MLflow points to a local file store by default and set the experiment.
    tracking_dir defaults to ./mlruns at project root if not provided.
    """
    try:
        uri = os.getenv("MLFLOW_TRACKING_URI")
        if uri:
            mlflow.set_tracking_uri(uri)
        else:
            base = get_project_root()
            default_dir = Path(tracking_dir) if tracking_dir else base / "mlruns"
            default_dir.mkdir(parents=True, exist_ok=True)
            mlflow.set_tracking_uri(default_dir.as_uri())
        mlflow.set_experiment(experiment)
    except Exception as e:
        logger.warning(f"MLflow setup warning: {e}")


def _flatten(prefix: str, d: Dict[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for k, v in d.items():
        kk = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            flat.update(_flatten(kk, v))
        else:
            flat[kk] = v
    return flat


def flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    return _flatten(prefix, d) if d else {}


def log_params_dict(d: Dict[str, Any], prefix: str = "") -> None:
    try:
        mlflow.log_params(flatten_dict(d, prefix))
    except Exception as e:
        logger.warning(f"Could not log params: {e}")


def log_metrics_dict(d: Dict[str, Any], prefix: str = "", step: Optional[int] = None) -> None:
    try:
        metrics = flatten_dict(d, prefix)
        if step is not None:
            mlflow.log_metrics(metrics, step=step)
        else:
            mlflow.log_metrics(metrics)
    except Exception as e:
        logger.warning(f"Could not log metrics: {e}")


def log_artifact_safe(path: Path | str) -> None:
    try:
        p = Path(path)
        if p.exists():
            mlflow.log_artifact(str(p))
    except Exception as e:
        logger.warning(f"Could not log artifact {path}: {e}")


# -----------------------
# Run-naming and tag utils
# -----------------------

def _run_git_cmd(args: list[str]) -> Optional[str]:
    try:
        res = subprocess.run(["git", *args], capture_output=True, text=True, check=True)
        return res.stdout.strip() or None
    except Exception:
        return None


def get_git_context() -> Dict[str, str]:
    commit = os.getenv("GITHUB_SHA") or _run_git_cmd(["rev-parse", "HEAD"]) or "unknown"
    branch = (
        os.getenv("GITHUB_REF_NAME")
        or os.getenv("GIT_BRANCH")
        or _run_git_cmd(["rev-parse", "--abbrev-ref", "HEAD"]) or "unknown"
    )
    dirty = _run_git_cmd(["status", "--porcelain"]) or ""
    is_dirty = "true" if dirty else "false"
    return {"git_commit": commit, "git_branch": branch, "git_dirty": is_dirty}


def pick_experiment_by_suffix(suffix: Optional[str]) -> str:
    s = (suffix or "").strip()
    if s == "":
        name = "mma_default"
    elif s == "symm":
        name = "mma_symm"
    elif s == "svd":
        name = "mma_svd"
    else:
        name = f"mma_{s}"
    try:
        mlflow.set_experiment(name)
    except Exception as e:
        logger.warning(f"Could not set experiment '{name}': {e}")
    return name


def make_run_name(stage: str, suffix: Optional[str], extra: Optional[str] = None) -> str:
    sfx = (suffix or "default").strip() or "default"
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    parts = [stage, sfx, ts]
    if extra:
        parts.append(extra)
    return "__".join(parts)


def start_run_with_tags(
    stage: str,
    suffix: Optional[str],
    run_type: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
    nested: Optional[bool] = None,
    run_name_extra: Optional[str] = None,
):
    """
    Convenience wrapper to start an MLflow run with consistent experiment naming and tags.
    Usage:
        with start_run_with_tags("split_trainval", suffix="", run_type="train"):
            mlflow.log_params({...})
            ...
    """
    # Ensure local tracking unless overridden; pick experiment
    try:
        exp = pick_experiment_by_suffix(suffix)
    except Exception:
        exp = None
    # Determine nesting automatically if not provided
    if nested is None:
        try:
            nested = mlflow.active_run() is not None
        except Exception:
            nested = False
    run_name = make_run_name(stage, suffix, run_name_extra)
    run = mlflow.start_run(run_name=run_name, nested=bool(nested))
    # Set tags
    try:
        ctx = get_git_context()
        base_tags: Dict[str, str] = {
            "stage": stage,
            "suffix": (suffix or ""),
            "run_type": (run_type or ""),
            "env": (os.getenv("CI", "false") == "true" and "ci" or "local"),
            **ctx,
        }
        if tags:
            base_tags.update(tags)
        mlflow.set_tags(base_tags)
    except Exception as e:
        logger.warning(f"Could not set MLflow tags: {e}")
    return run
