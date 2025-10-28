"""
使用 Optuna 对 GraphMAE 模型进行超参数搜索。
"""

import argparse
import copy
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict

import optuna
import numpy as np

from autoencoder_utils import load_config, plot_training_curves
from train_graphmae import train_graphmae


def prepare_config_for_trial(base_config: Dict[str, Any], trial: optuna.Trial, tmp_dir: Path) -> Dict[str, Any]:
    config = copy.deepcopy(base_config)

    model_cfg = config["model"]
    train_cfg = config["training"]
    sched_cfg = config["optimizer"]["scheduler_params"]
    loss_cfg = config.get("loss", {})

    model_cfg["dropout"] = trial.suggest_float("dropout", 0.1, 0.5)
    model_cfg["num_layers"] = trial.suggest_int("num_layers", 2, 4)
    model_cfg["hidden_dim"] = trial.suggest_int("hidden_dim", 128, 320, step=32)
    model_cfg["embedding_dim"] = trial.suggest_int("embedding_dim", 64, 192, step=32)
    model_cfg["conv_type"] = trial.suggest_categorical("conv_type", ["gcn", "gat", "sage"])
    model_cfg["mask_ratio"] = trial.suggest_float("mask_ratio", 0.05, 0.3)

    train_cfg["learning_rate"] = trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True)
    train_cfg["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    train_cfg["num_epochs"] = trial.suggest_int("num_epochs", 40, 100)
    train_cfg["link_weight"] = trial.suggest_float("link_weight", 1e-3, 0.2, log=True)

    loss_cfg["feature_weight"] = 1.0
    loss_cfg["link_weight"] = train_cfg["link_weight"]
    config["loss"] = loss_cfg

    sched_cfg["factor"] = trial.suggest_float("scheduler_factor", 0.2, 0.8)
    sched_cfg["patience"] = trial.suggest_int("scheduler_patience", 3, 12)

    paths_cfg = config["paths"]
    for key, value in list(paths_cfg.items()):
        if key == "data":
            continue
        paths_cfg[key] = str(tmp_dir / Path(value).name)

    return config


def objective(trial: optuna.Trial) -> float:
    base_config = load_config(Path("models/config/graphmae_config.json"))

    tmp_dir = Path(tempfile.mkdtemp(prefix=f"graphmae_trial_{trial.number}_"))
    history_path = tmp_dir / "training_history.npy"
    metrics_path = tmp_dir / "best_metrics.json"
    best_total_loss = float("inf")
    best_feature_loss = float("inf")
    best_link_loss = float("inf")
    best_epoch = None
    metrics_at_best = {}
    try:
        trial_config = prepare_config_for_trial(base_config, trial, tmp_dir)
        result = train_graphmae(trial_config, trial_mode=True)
        best_auc = result.get("best_val_auc", 0.0)

        history = result.get("training_history", {})
        if history:
            np.save(history_path, history)
            trial.set_user_attr("training_history", history)
            total_loss_history = history.get("val_loss", [])
            feature_loss_history = history.get("val_feature_loss", [])
            link_loss_history = history.get("val_link_loss", [])
            if total_loss_history:
                total_array = np.array(total_loss_history, dtype=float)
                finite_mask = np.isfinite(total_array)
                if finite_mask.any():
                    safe_total = total_array.copy()
                    safe_total[~finite_mask] = np.nan
                    best_epoch = int(np.nanargmin(safe_total))
                    best_total_loss = float(safe_total[best_epoch])
            if feature_loss_history and best_epoch is not None:
                feature_array = np.array(feature_loss_history, dtype=float)
                if len(feature_array) > best_epoch and np.isfinite(feature_array[best_epoch]):
                    best_feature_loss = float(feature_array[best_epoch])
            elif feature_loss_history:
                feature_array = np.array(feature_loss_history, dtype=float)
                finite_mask = np.isfinite(feature_array)
                if finite_mask.any():
                    best_feature_loss = float(np.nanmin(np.where(finite_mask, feature_array, np.nan)))
            if link_loss_history and best_epoch is not None:
                link_array = np.array(link_loss_history, dtype=float)
                if len(link_array) > best_epoch and np.isfinite(link_array[best_epoch]):
                    best_link_loss = float(link_array[best_epoch])
            elif link_loss_history:
                link_array = np.array(link_loss_history, dtype=float)
                finite_mask = np.isfinite(link_array)
                if finite_mask.any():
                    best_link_loss = float(np.nanmin(np.where(finite_mask, link_array, np.nan)))
            if best_epoch is not None:
                for key in ("val_precision", "val_accuracy", "val_auc", "val_ap"):
                    series = history.get(key, [])
                    if len(series) > best_epoch and np.isfinite(series[best_epoch]):
                        metrics_at_best[key] = float(series[best_epoch])
                if np.isfinite(best_total_loss):
                    metrics_at_best["val_loss"] = best_total_loss
                if best_feature_loss is not None and np.isfinite(best_feature_loss):
                    metrics_at_best["val_feature_loss"] = best_feature_loss
                if best_link_loss is not None and np.isfinite(best_link_loss):
                    metrics_at_best["val_link_loss"] = best_link_loss
        best_metrics = {
            "best_val_metrics": result.get("best_val_metrics", {}),
            "best_val_auc": best_auc,
            "best_total_loss": best_total_loss,
            "best_feature_loss": best_feature_loss,
            "best_link_loss": best_link_loss,
            "best_epoch": best_epoch,
            "metrics_at_best_loss": metrics_at_best
        }
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(best_metrics, f, ensure_ascii=False, indent=2)
        trial.set_user_attr("metrics", best_metrics)
        trial.set_user_attr("best_val_auc", best_auc)
        trial.set_user_attr("best_total_loss", best_total_loss)
        trial.set_user_attr("best_feature_loss", best_feature_loss)
        trial.set_user_attr("best_link_loss", best_link_loss)
        trial.set_user_attr("best_epoch", best_epoch)
        trial.set_user_attr("metrics_at_best_loss", metrics_at_best)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if not np.isfinite(best_total_loss):
        return float("inf")
    return best_total_loss


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna 调参 - GraphMAE")
    parser.add_argument("--trials", type=int, default=20, help="试验次数")
    parser.add_argument("--timeout", type=int, default=None, help="调参最长时间（秒）")
    parser.add_argument("--storage", type=str, default=None, help="Optuna storage，例如 sqlite:///graphmae_optuna.db")
    parser.add_argument("--study-name", type=str, default="graphmae-optuna-study", help="Study 名称")
    args = parser.parse_args()

    study_kwargs = {"study_name": args.study_name, "direction": "minimize"}
    if args.storage:
        study_kwargs["storage"] = args.storage
        study_kwargs["load_if_exists"] = True

    study = optuna.create_study(**study_kwargs)
    study.optimize(objective, n_trials=args.trials, timeout=args.timeout)

    print("=== Optuna 最佳结果 (GraphMAE) ===")
    best_total = study.best_value
    if np.isfinite(best_total):
        print(f"最佳验证总损失(MSE+BCE): {best_total:.6f}")
    else:
        print("最佳验证总损失(MSE+BCE): 未获取有效值")
    print("最佳参数:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")

    best_history = study.best_trial.user_attrs.get("training_history")
    best_epoch = study.best_trial.user_attrs.get("best_epoch")
    if best_epoch is not None:
        print(f"对应最佳轮次(基于总损失): {best_epoch + 1}")
    ref_auc = study.best_trial.user_attrs.get("best_val_auc")
    if ref_auc is not None:
        print(f"参考验证AUC: {ref_auc:.4f}")
    feature_loss = study.best_trial.user_attrs.get("best_feature_loss")
    if feature_loss is not None and np.isfinite(feature_loss):
        print(f"对应验证特征重建MSE: {feature_loss:.6f}")
    link_loss = study.best_trial.user_attrs.get("best_link_loss")
    if link_loss is not None and np.isfinite(link_loss):
        print(f"对应验证链接重建BCE: {link_loss:.6f}")
    ref_metrics = study.best_trial.user_attrs.get("metrics_at_best_loss", {})
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
        curve_path = Path("models/outputs/graphmae_optuna_best_curves.png")
        plot_training_curves(best_history, str(curve_path), model_name="GraphMAE (Optuna Best)")
        print(f"最佳训练曲线已保存至: {curve_path}")
    else:
        print("未找到最佳 trial 的训练历史，无法绘制曲线。")


if __name__ == "__main__":
    main()
