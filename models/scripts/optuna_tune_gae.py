"""
使用 Optuna 对 GAE 模型进行超参数搜索。
"""

import argparse
import copy
import json
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Any

import optuna
import numpy as np

from autoencoder_utils import load_config, plot_training_curves
from train_gae import train_gae


def prepare_config_for_trial(base_config: Dict[str, Any], trial: optuna.Trial, tmp_dir: Path) -> Dict[str, Any]:
    """复制配置并根据 trial 建议更新关键超参数，同时把输出路径指向临时目录。"""
    config = copy.deepcopy(base_config)

    model_cfg = config["model"]
    train_cfg = config["training"]
    sched_cfg = config["optimizer"]["scheduler_params"]

    model_cfg["dropout"] = trial.suggest_float("dropout", 0.1, 0.5)
    train_cfg["learning_rate"] = trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True)
    train_cfg["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    sched_cfg["factor"] = trial.suggest_float("scheduler_factor", 0.2, 0.8)
    sched_cfg["patience"] = trial.suggest_int("scheduler_patience", 3, 12)

    # 训练轮数适当缩短，提升调参效率
    train_cfg["num_epochs"] = trial.suggest_int("num_epochs", 40, 100)

    # 调整模型结构参数（可选）
    model_cfg["num_layers"] = trial.suggest_int("num_layers", 2, 4, step=1)
    model_cfg["hidden_dim"] = trial.suggest_int("hidden_dim", 128, 320, step=32)
    model_cfg["embedding_dim"] = trial.suggest_int("embedding_dim", 64, 192, step=32)
    model_cfg["conv_type"] = trial.suggest_categorical("conv_type", ["gcn", "gat", "sage"])

    # 将所有输出文件写入临时目录，避免污染主目录
    paths_cfg = config["paths"]
    for key, value in list(paths_cfg.items()):
        if key == "data":
            # data 路径保持绝对路径
            paths_cfg[key] = value
            continue
        original_name = Path(value).name
        paths_cfg[key] = str(tmp_dir / original_name)

    return config


def objective(trial: optuna.Trial) -> float:
    base_config = load_config(Path("models/config/gae_config.json"))

    tmp_dir = Path(tempfile.mkdtemp(prefix=f"gae_trial_{trial.number}_"))
    history_path = tmp_dir / "training_history.npy"
    metrics_path = tmp_dir / "best_metrics.json"
    best_val_loss = float("inf")
    best_epoch = None
    metrics_at_best = {}
    try:
        trial_config = prepare_config_for_trial(base_config, trial, tmp_dir)
        result = train_gae(trial_config, trial_mode=True)
        best_auc = result.get("best_val_auc", 0.0)

        # 持久化历史与指标，方便后续复现曲线
        history = result.get("training_history", {})
        if history:
            np.save(history_path, history)
            trial.set_user_attr("history_path", str(history_path))
            trial.set_user_attr("training_history", history)
            val_loss_history = history.get("val_loss", [])
            if val_loss_history:
                val_loss_array = np.array(val_loss_history, dtype=float)
                finite_mask = np.isfinite(val_loss_array)
                if finite_mask.any():
                    safe_losses = val_loss_array.copy()
                    safe_losses[~finite_mask] = np.nan
                    best_epoch = int(np.nanargmin(safe_losses))
                    best_val_loss = float(safe_losses[best_epoch])
                    for key in ("val_precision", "val_accuracy", "val_auc", "val_ap"):
                        series = history.get(key, [])
                        if len(series) > best_epoch:
                            metrics_at_best[key] = float(series[best_epoch])
                    if np.isfinite(best_val_loss):
                        metrics_at_best["val_loss"] = best_val_loss
        best_metrics = {
            "best_val_metrics": result.get("best_val_metrics", {}),
            "best_val_auc": best_auc,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "metrics_at_best_loss": metrics_at_best
        }
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(best_metrics, f, ensure_ascii=False, indent=2)
        trial.set_user_attr("metrics_path", str(metrics_path))
        trial.set_user_attr("best_val_auc", best_auc)
        trial.set_user_attr("best_val_metrics", best_metrics)
        trial.set_user_attr("best_val_loss", best_val_loss)
        trial.set_user_attr("best_epoch", best_epoch)
        trial.set_user_attr("metrics_at_best_loss", metrics_at_best)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if not np.isfinite(best_val_loss):
        return float("inf")
    return best_val_loss


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna 调参 - GAE")
    parser.add_argument("--trials", type=int, default=20, help="试验次数")
    parser.add_argument("--timeout", type=int, default=None, help="调参最长时间（秒）")
    parser.add_argument("--storage", type=str, default=None, help="Optuna storage，例如 sqlite:///gae_optuna.db")
    parser.add_argument("--study-name", type=str, default="gae-optuna-study", help="Study 名称")
    args = parser.parse_args()

    study_kwargs = {"study_name": args.study_name, "direction": "minimize"}
    if args.storage:
        study_kwargs["storage"] = args.storage
        study_kwargs["load_if_exists"] = True

    study = optuna.create_study(**study_kwargs)
    study.optimize(objective, n_trials=args.trials, timeout=args.timeout)

    print("=== Optuna 最佳结果 ===")
    best_metric_value = study.best_value
    if np.isfinite(best_metric_value):
        print(f"最佳验证BCE: {best_metric_value:.6f}")
    else:
        print("最佳验证BCE: 未获取有效值")
    print("最佳参数:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")

    # 加载最佳 trial 的历史，绘制曲线
    best_trial = study.best_trial
    best_history = best_trial.user_attrs.get("training_history")
    best_epoch = best_trial.user_attrs.get("best_epoch")
    if best_epoch is not None:
        print(f"对应最佳轮次(基于验证BCE): {best_epoch + 1}")
    ref_auc = best_trial.user_attrs.get("best_val_auc")
    if ref_auc is not None:
        print(f"参考验证AUC: {ref_auc:.4f}")
    ref_metrics = best_trial.user_attrs.get("metrics_at_best_loss", {})
    if ref_metrics:
        precision = ref_metrics.get("val_precision")
        accuracy = ref_metrics.get("val_accuracy")
        auc = ref_metrics.get("val_auc")
        ap = ref_metrics.get("val_ap")
        if precision is not None:
            print(f"对应精度: {precision:.4f}")
        if accuracy is not None:
            print(f"对应准确率: {accuracy:.4f}")
        if auc is not None:
            print(f"对应AUC: {auc:.4f}")
        if ap is not None:
            print(f"对应平均精度: {ap:.4f}")

    if best_history:
        curve_path = Path("models/outputs/gae_optuna_best_curves.png")
        plot_training_curves(best_history, str(curve_path), model_name="GAE (Optuna Best)")
        print(f"最佳训练曲线已保存至: {curve_path}")
    else:
        print("未找到最佳 trial 的训练历史，无法绘制曲线。")


if __name__ == "__main__":
    main()
