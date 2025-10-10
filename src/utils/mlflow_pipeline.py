"""
Modern MLflow Pipeline Management for MMA Project

This module provides a clean, hierarchical MLflow tracking structure that properly
organizes the 3-stage MMA pipeline:

1. Data Cleaning Stage
2. Feature Engineering & Dataset Preparation Stage  
3. Model Training & Prediction Stage

Key Features:
- Single experiment per pipeline run with nested child runs for each stage
- Proper parent-child relationships between pipeline stages
- Consistent tagging and naming conventions
- Model registry integration when available
- Pipeline-level metadata tracking
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import os
import subprocess
from datetime import datetime
import mlflow
from mlflow.tracking import MlflowClient
from loguru import logger
from contextlib import contextmanager

from .general import get_project_root


class MLflowPipelineManager:
    """
    Manages MLflow tracking for the MMA prediction pipeline.
    
    Creates a hierarchical structure:
    Pipeline Run (parent)
    ├── Stage 1: Data Cleaning (child)
    ├── Stage 2: Feature Engineering (child)  
    │   ├── Merge Features (nested child)
    │   ├── Split Train/Val (nested child)
    │   ├── Build Prediction Set (nested child)
    │   └── Transform Data (nested child)
    └── Stage 3: Model Training (child)
        ├── Hyperparameter Optimization (nested child)
        ├── Feature Selection (nested child)
        └── Final Predictions (nested child)
    """
    
    def __init__(self, 
                 experiment_name: Optional[str] = None,
                 tracking_uri: Optional[str] = None,
                 pipeline_name: Optional[str] = None):
        """
        Initialize the MLflow pipeline manager.
        
        Args:
            experiment_name: Name of the MLflow experiment. Defaults to timestamped name.
            tracking_uri: MLflow tracking URI. Defaults to local ./mlruns
            pipeline_name: Human readable pipeline name for run naming
        """
        self.pipeline_name = pipeline_name or "mma_pipeline"
        self.setup_tracking(tracking_uri, experiment_name)
        self.pipeline_run_id: Optional[str] = None
        self.current_stage_run_id: Optional[str] = None
        self._git_context = self._get_git_context()
        
    def setup_tracking(self, tracking_uri: Optional[str] = None, 
                      experiment_name: Optional[str] = None) -> None:
        """Setup MLflow tracking URI and experiment."""
        # Set tracking URI
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            # Default to local file store
            repo_root = get_project_root()
            mlruns_dir = repo_root / "mlruns"
            mlruns_dir.mkdir(parents=True, exist_ok=True)
            mlflow.set_tracking_uri(mlruns_dir.as_uri())
            
        # Set experiment name
        if not experiment_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"mma_pipeline_{timestamp}"
            
        try:
            mlflow.set_experiment(experiment_name)
            self.experiment_name = experiment_name
            logger.info(f"Using MLflow experiment: {experiment_name}")
        except Exception as e:
            logger.error(f"Failed to set MLflow experiment '{experiment_name}': {e}")
            # Fallback to default
            mlflow.set_experiment("Default")
            self.experiment_name = "Default"
    
    @contextmanager
    def pipeline_run(self, 
                    suffix: str = "",
                    feature_sets: Optional[Dict[str, Any]] = None,
                    **pipeline_params):
        """
        Context manager for the entire pipeline run (parent run).
        
        Args:
            suffix: Data transformation suffix ("", "symm", "svd")
            feature_sets: Dictionary of feature sets being used
            **pipeline_params: Additional pipeline-level parameters
        """
        run_name = self._make_pipeline_run_name(suffix)
        
        # Start parent pipeline run
        run = mlflow.start_run(run_name=run_name)
        self.pipeline_run_id = run.info.run_id
        
        try:
            # Log pipeline-level parameters and tags
            self._log_pipeline_metadata(suffix, feature_sets, pipeline_params)
            logger.info(f"Started MLflow pipeline run: {run_name}")
            yield self
            
            # Mark pipeline as successful
            mlflow.set_tag("pipeline_status", "completed")
            logger.info(f"Completed MLflow pipeline run: {run_name}")
            
        except Exception as e:
            # Mark pipeline as failed
            mlflow.set_tag("pipeline_status", "failed")
            mlflow.set_tag("error_message", str(e))
            logger.error(f"Pipeline run failed: {e}")
            raise
        finally:
            mlflow.end_run()
            self.pipeline_run_id = None
    
    @contextmanager  
    def stage_run(self, 
                 stage_name: str,
                 stage_type: str = "processing",
                 **stage_params):
        """
        Context manager for a pipeline stage run (child of pipeline run).
        
        Args:
            stage_name: Name of the stage ("data_cleaning", "feature_engineering", "model_training")  
            stage_type: Type of stage ("processing", "training", "prediction")
            **stage_params: Stage-specific parameters
        """
        if not self.pipeline_run_id:
            raise RuntimeError("Must be called within a pipeline_run context")
            
        run_name = f"{stage_name}_{datetime.now().strftime('%H%M%S')}"
        
        # Start nested child run
        run = mlflow.start_run(run_name=run_name, nested=True)
        self.current_stage_run_id = run.info.run_id
        
        try:
            # Log stage metadata
            mlflow.set_tags({
                "stage": stage_name,
                "stage_type": stage_type,
                "pipeline_run_id": self.pipeline_run_id,
                **self._git_context
            })
            
            if stage_params:
                mlflow.log_params(self._flatten_params(stage_params, "stage"))
                
            logger.info(f"Started stage: {stage_name}")
            yield self
            
            mlflow.set_tag("stage_status", "completed")
            logger.info(f"Completed stage: {stage_name}")
            
        except Exception as e:
            mlflow.set_tag("stage_status", "failed") 
            mlflow.set_tag("stage_error", str(e))
            logger.error(f"Stage {stage_name} failed: {e}")
            raise
        finally:
            mlflow.end_run()
            self.current_stage_run_id = None
    
    @contextmanager
    def substage_run(self, 
                    substage_name: str,
                    **substage_params):
        """
        Context manager for a substage run (nested child of stage run).
        
        Args:
            substage_name: Name of the substage ("merge_features", "cv_optimization", etc.)
            **substage_params: Substage-specific parameters
        """
        if not self.current_stage_run_id:
            raise RuntimeError("Must be called within a stage_run context")
            
        run_name = f"{substage_name}_{datetime.now().strftime('%H%M%S')}"
        
        # Start nested child run  
        run = mlflow.start_run(run_name=run_name, nested=True)
        
        try:
            # Log substage metadata
            mlflow.set_tags({
                "substage": substage_name,
                "stage_run_id": self.current_stage_run_id,
                "pipeline_run_id": self.pipeline_run_id,
                **self._git_context
            })
            
            if substage_params:
                mlflow.log_params(self._flatten_params(substage_params, "substage"))
                
            logger.debug(f"Started substage: {substage_name}")
            yield self
            
            mlflow.set_tag("substage_status", "completed")
            logger.debug(f"Completed substage: {substage_name}")
            
        except Exception as e:
            mlflow.set_tag("substage_status", "failed")
            mlflow.set_tag("substage_error", str(e))
            logger.error(f"Substage {substage_name} failed: {e}")
            raise
        finally:
            mlflow.end_run()
    
    def log_data_artifact(self, 
                         file_path: Union[str, Path],
                         artifact_type: str = "data") -> None:
        """Log a data artifact with consistent naming."""
        try:
            path = Path(file_path)
            if path.exists():
                mlflow.log_artifact(str(path))
                mlflow.set_tag(f"{artifact_type}_artifact", path.name)
                logger.debug(f"Logged {artifact_type} artifact: {path.name}")
        except Exception as e:
            logger.warning(f"Could not log artifact {file_path}: {e}")
    
    def log_model_with_metadata(self, 
                               model,
                               model_name: str,
                               signature: Optional[Any] = None,
                               input_example: Optional[Any] = None,
                               **model_metadata) -> None:
        """Log model with comprehensive metadata."""
        try:
            # Log the model
            mlflow.sklearn.log_model(
                model, 
                model_name,
                signature=signature,
                input_example=input_example
            )
            
            # Log model metadata
            if model_metadata:
                mlflow.log_params(self._flatten_params(model_metadata, "model"))
                
            mlflow.set_tag("model_logged", "true")
            logger.info(f"Logged model: {model_name}")
            
        except Exception as e:
            logger.warning(f"Could not log model {model_name}: {e}")
    
    def log_metrics_safe(self, 
                        metrics: Dict[str, Any], 
                        prefix: str = "",
                        step: Optional[int] = None) -> None:
        """Log metrics with error handling."""
        try:
            flat_metrics = self._flatten_params(metrics, prefix)
            if step is not None:
                mlflow.log_metrics(flat_metrics, step=step)
            else:
                mlflow.log_metrics(flat_metrics)
        except Exception as e:
            logger.warning(f"Could not log metrics: {e}")
    
    def log_params_safe(self, 
                       params: Dict[str, Any], 
                       prefix: str = "") -> None:
        """Log parameters with error handling."""
        try:
            flat_params = self._flatten_params(params, prefix)
            mlflow.log_params(flat_params)
        except Exception as e:
            logger.warning(f"Could not log params: {e}")
    
    def _make_pipeline_run_name(self, suffix: str) -> str:
        """Generate consistent pipeline run name."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix_part = f"_{suffix}" if suffix else ""
        return f"{self.pipeline_name}{suffix_part}_{timestamp}"
    
    def _log_pipeline_metadata(self, 
                              suffix: str,
                              feature_sets: Optional[Dict[str, Any]],
                              pipeline_params: Dict[str, Any]) -> None:
        """Log comprehensive pipeline metadata."""
        # Pipeline-level tags
        tags = {
            "pipeline_name": self.pipeline_name,
            "data_suffix": suffix or "raw",
            "environment": "ci" if os.getenv("CI") == "true" else "local",
            "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}",
            **self._git_context
        }
        mlflow.set_tags(tags)
        
        # Pipeline-level parameters
        params = {
            "pipeline.suffix": suffix or "raw",
            "pipeline.start_time": datetime.now().isoformat(),
            **pipeline_params
        }
        
        if feature_sets:
            params.update(self._flatten_params(feature_sets, "features"))
            
        mlflow.log_params(params)
    
    def _flatten_params(self, params: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        """Flatten nested parameter dictionaries."""
        flat = {}
        for key, value in params.items():
            new_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                flat.update(self._flatten_params(value, new_key))
            elif isinstance(value, (list, tuple)):
                flat[new_key] = str(value)
            else:
                flat[new_key] = value
        return flat
    
    def _get_git_context(self) -> Dict[str, str]:
        """Get git context for reproducibility."""
        def run_git_cmd(args: List[str]) -> Optional[str]:
            try:
                result = subprocess.run(
                    ["git"] + args, 
                    capture_output=True, 
                    text=True, 
                    check=True
                )
                return result.stdout.strip() or None
            except Exception:
                return None
        
        commit = (os.getenv("GITHUB_SHA") or 
                 run_git_cmd(["rev-parse", "HEAD"]) or 
                 "unknown")
        branch = (os.getenv("GITHUB_REF_NAME") or
                 os.getenv("GIT_BRANCH") or  
                 run_git_cmd(["rev-parse", "--abbrev-ref", "HEAD"]) or
                 "unknown")
        dirty = run_git_cmd(["status", "--porcelain"]) or ""
        
        return {
            "git_commit": commit,
            "git_branch": branch, 
            "git_dirty": "true" if dirty else "false"
        }


# Convenience functions for backward compatibility
_pipeline_manager: Optional[MLflowPipelineManager] = None

def get_pipeline_manager() -> Optional[MLflowPipelineManager]:
    """Get the current pipeline manager instance."""
    return _pipeline_manager

def set_pipeline_manager(manager: MLflowPipelineManager) -> None:
    """Set the current pipeline manager instance."""
    global _pipeline_manager
    _pipeline_manager = manager

def pipeline_run(**kwargs):
    """Convenience function for pipeline run context."""
    if _pipeline_manager is None:
        raise RuntimeError("No pipeline manager set. Call set_pipeline_manager() first.")
    return _pipeline_manager.pipeline_run(**kwargs)

def stage_run(stage_name: str, **kwargs):
    """Convenience function for stage run context.""" 
    if _pipeline_manager is None:
        raise RuntimeError("No pipeline manager set. Call set_pipeline_manager() first.")
    return _pipeline_manager.stage_run(stage_name, **kwargs)

def substage_run(substage_name: str, **kwargs):
    """Convenience function for substage run context."""
    if _pipeline_manager is None:
        raise RuntimeError("No pipeline manager set. Call set_pipeline_manager() first.")
    return _pipeline_manager.substage_run(substage_name, **kwargs)