import argparse
import json
import os
import re
import sys
import pickle
import itertools
from pathlib import Path
from turtle import home
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score

sys.path.append(str(Path(Path.home(),'scripts_generales'))) if 'Users/gp' in str(Path.home()) else sys.path.append(str(Path(Path.home(),'gonza','scripts_generales')))

import utils
# ---------------------- Helpers genéricos ---------------------- #

def _to_numpy(a: Any) -> np.ndarray:
    if isinstance(a, np.ndarray):
        return a
    if hasattr(a, "values"):  # pandas Series/DataFrame
        return a.values
    return np.asarray(a)

def _safe_load_pickle(p: Path) -> Any:
    with p.open("rb") as f:
        return pickle.load(f)

def _extract_scores(outputs: Any, stage: str, n: int) -> Optional[np.ndarray]:
    """
    Intenta extraer probabilidades/scores compatibles con ROC de varias formas posibles.
    Devuelve un array (n,) para binario o (n, C) para multiclase.
    """
    cands = []

    # Formatos comunes en dict
    if isinstance(outputs, dict):
        keys = [
            f"proba_{stage}", f"probs_{stage}", f"prob_{stage}", f"y_proba_{stage}",
            f"scores_{stage}", f"score_{stage}", f"y_score_{stage}",
            f"decision_{stage}", f"decision_function_{stage}",
            "proba", "probs", "prob", "y_proba",
            "scores", "score", "y_score",
            "decision", "decision_function",
            stage,
        ]
        for k in keys:
            if k in outputs:
                cands.append(outputs[k])
        if stage in outputs and isinstance(outputs[stage], dict):
            for k in keys:
                if k in outputs[stage]:
                    cands.append(outputs[stage][k])
        # a veces listas/tuplas con arrays
        for v in outputs.values():
            if isinstance(v, (list, tuple)):
                cands.extend(v)

    if isinstance(outputs, (list, tuple)):
        cands.extend(outputs)

    good = []
    for c in cands:
        try:
            arr = _to_numpy(c)
            if arr.ndim == 1 and arr.shape[0] == n:
                good.append(arr)
            elif arr.ndim == 2 and arr.shape[0] == n:
                good.append(arr)
        except Exception:
            pass

    if not good:
        return None

    # Preferimos probabilidades multiclase (n,C) frente a 1D
    good = sorted(good, key=lambda a: (a.ndim, a.shape[1] if a.ndim == 2 else 1), reverse=True)
    return good[0]

def _binarize_multiclass(y_true: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    classes = sorted(np.unique(y_true).tolist())
    Y = np.zeros((y_true.shape[0], len(classes)), dtype=int)
    idx = {c: i for i, c in enumerate(classes)}
    for i, c in enumerate(y_true):
        Y[i, idx[c]] = 1
    return Y, classes

def _guess_pos_col(y_true: np.ndarray, proba2d: np.ndarray) -> int:
    labels = np.unique(y_true)
    if proba2d.shape[1] == 2 and set(labels.tolist()) == {0,1}:
        return 1
    return proba2d.shape[1]-1  # última columna por convención

def _build_results_dir_from_config(config: Dict[str, Any]) -> Path:
    home = Path(os.environ.get("HOME", Path.home()))
    if "Users/gp" in str(home):
        return home / "results" / config["project_name"]
    else:
        return Path("D:/CNC_Audio/gonza/results", config["project_name"])

def _list_dimensions(results_dir: Path, task: str) -> List[str]:
    p = results_dir / task
    if not p.exists():
        return []
    return [d.name for d in p.iterdir() if d.is_dir()]

def _path_leaf(results_dir: Path, task: str, dimension: str, config: Dict[str, Any], y_label: str) -> Path:
    scaler_name = config['scaler_name']
    kfold_folder = config['kfold_folder']
    stat_folder  = config['stat_folder']
    scoring      = config['scoring_metric']
    hyp_opt      = True if int(config['n_iter']) > 0 else False
    feature_sel  = bool(config['feature_selection'])
    shuffle_lab  = bool(config['shuffle_labels'])

    sub = [
        task, dimension, scaler_name, kfold_folder, y_label, stat_folder, "bayes", scoring,
        "hyp_opt" if hyp_opt else "",
        "feature_selection" if feature_sel else "",
        "shuffle" if shuffle_lab else "",
    ]
    return results_dir.joinpath(*[s for s in sub if s])

# ---------------------- Lógica principal ---------------------- #

def main():
    # Cargar config y main_config como en tu pipeline
    here = Path(__file__).parent
    config = json.load((here / "config.json").open())

    project_name = config["project_name"]
    scoring = config["scoring_metric"]
    kfold_folder = config["kfold_folder"]
    scaler_name = config["scaler_name"]
    stat_folder = config["stat_folder"]
    bootstrap_method = config["bootstrap_method"]
    hyp_opt = bool(config["n_iter"] > 0)
    feature_selection = bool(config["feature_selection"])

    if "Users/gp" in str(home):
        save_dir = home / 'results' / project_name / 'rocs'
    else:
        save_dir = Path("D:/CNC_Audio/gonza/results", project_name,'rocs')

    main_config = json.load((here / "main_config.json").open())

    tasks = main_config["tasks"][project_name]
    y_labels_cfg = main_config["y_labels"][project_name]
    test_size = main_config["test_size"][project_name]

    results_dir = _build_results_dir_from_config(config)

    # Si se usa --only-best, localizamos el CSV de mejores
    best_csv = f"best_best_models_{scoring}_{kfold_folder}_{scaler_name}_{stat_folder}_{bootstrap_method}_hyp_opt_feature_selection_bayes.csv".replace("__","_")

    if not hyp_opt:
        best_csv = best_csv.replace("_hyp_opt", "")
    if not feature_selection:
        best_csv = best_csv.replace("_feature_selection", "")

    best_df = pd.read_csv(Path(results_dir,best_csv))

    save_dir.mkdir(parents=True, exist_ok=True)

    # Recorrer como en tu script
    for task in tasks:
        dims = _list_dimensions(results_dir, task)
        if isinstance(y_labels_cfg, dict):
            y_labels = y_labels_cfg[task]
        else:
            y_labels = y_labels_cfg

        for dimension, y_label in itertools.product(dims, y_labels):
            leaf = _path_leaf(results_dir, task, dimension, config, y_label)
            if not leaf.exists():
                continue

            # Semillas
            if float(test_size) > 0:
                seeds = [d.name for d in leaf.iterdir() if d.is_dir() and "random_seed" in d.name]
                if not seeds:
                    seeds = ["random_seed_0"]
            else:
                seeds = [""]  # sin carpeta de seed

            for seed in seeds:
                combo_dir = leaf / seed if seed else leaf
                if not combo_dir.exists():
                    continue

                sel = best_df[
                    (best_df["task"]==task) &
                    (best_df["dimension"]==dimension) &
                    (best_df["y_label"]==y_label)
                ]
                if not sel.empty:
                    model_type = sel["model_type"].unique().tolist()[0]

                try:
                    outputs_tmp, y_vec = utils._load_data(
                        results_dir, task, dimension, y_label, model_type,
                        seed, config, bayes=True, scoring=scoring
                    )
                except Exception as e:
                    print(f"[WARN] No se pudo cargar datos para {task}/{dimension}/{y_label}/{seed}: {e}")
                    continue

                y_true = _to_numpy(y_vec)
                classes = np.unique(y_true)
                is_binary = classes.size == 2

                # ---- Plot ----
                plt.figure(figsize=(8,6))

                try:
                    outputs, y_check = utils._load_data(
                        results_dir, task, dimension, y_label, model_type,
                        seed, config, bayes=True, scoring=scoring
                    )
                except Exception as e:
                    print(f"[WARN] _load_data falló para {model_type} en {task}/{dimension}/{y_label}/{seed}: {e}")
                    continue

                scores = _to_numpy(outputs)
                
                fpr_grid = np.linspace(0, 1, 100)
                tpr_mat = []  # TPR interpolados por seed (binario o micro)
                aucs = []
                
                if is_binary:
                    for r in range(y_true.shape[0]):

                        fpr, tpr, _ = roc_curve(y_true[r], scores[r,:,1], pos_label=classes.max())
                        tpr_interp = np.interp(fpr_grid, fpr, tpr)  # interpola TPR en el grid común

                        tpr_mat.append(tpr_interp)
                        aucs.append(auc(fpr, tpr))
                    
                    tpr_mat = np.vstack(tpr_mat)

                    plt.plot(fpr_grid, tpr_mat.mean(axis=0), label=f"{model_type} (AUC={np.mean(aucs):.3f})")
                
                # Línea diagonal
                plt.plot([0,1],[0,1], linestyle="--")
                title = f"ROC — task={task} | dim={dimension} | y={y_label} | seed={seed or 'none'}"
                plt.title(title)
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.legend(loc="lower right")
                plt.tight_layout()

                out_name = f"roc_dev_{task}__{dimension}__{y_label}__{(seed or 'none').replace('random_seed_','rs')}.png"
                out_path = Path(save_dir) / out_name
                out_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(out_path, dpi=300, bbox_inches="tight")
                plt.close()
                print("[OK] Guardado:", out_path)

if __name__ == "__main__":
    main()