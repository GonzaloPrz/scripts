# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 22:50:13 2025

@author: Agus
"""
import os, re, time
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import QuantileTransformer, PowerTransformer, StandardScaler, RobustScaler
from sklearn.decomposition import PCA

# ===================== CONFIG =====================
BASE_CSV_PATH   = r"D:/LAB/Becarios/REDLAT/REDLAT_features_subset.csv"       # CSV original (header en 2 líneas)
MACRO_MAP_PATH  = r"D:/LAB/Becarios/REDLAT/feature_macrodomains_map.csv"     # mapping generado previamente
OUT_DIR         = r"D:/LAB/Becarios/REDLAT/data/pca_by_macrodomain"          # carpeta de salida

ID_COL          = "id"           # nombre del índice (en el CSV)
GROUP_BY_MODALITY = False        # True => PCA por (macro_domain × modality). False => solo por macro_domain.

# PCA
VAR_TARGET_EXPORT       = 0.85
MIN_COMPONENTS_EXPORT   = 2
MAX_COMPONENTS_EXPORT   = 30

# Preproc
TRANSFORM_METHOD        = "rankgauss"   # "rankgauss" | "yeojohnson"
ROBUST_SCALE            = False         # False=StandardScaler | True=RobustScaler
WINSORIZE               = True
CLIP_PCTS               = (0.5, 99.5)

# Redundancia
APPLY_REDUNDANCY_FILTER_EXPORT = True
REDUNDANCY_THRESHOLD           = 0.98

# Archivo merged
MERGED_PC_FILENAME     = "ALL_MACROS_PC_scores.csv"
RANDOM_STATE = 42
# ==================================================

os.makedirs(OUT_DIR, exist_ok=True)
LOG_PATH = os.path.join(OUT_DIR, "run_log.txt")

def ts(): return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
def log(msg):
    line = f"[{ts()}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")

with open(LOG_PATH, "w", encoding="utf-8") as f:
    f.write(f"[{ts()}] Run started. OUT_DIR={OUT_DIR}\n")

# ---------- 1) Leer CSV base con fix de encabezado ----------
from io import StringIO

def read_csv_with_header_fix(path, encoding="utf-8"):
    with open(path, "r", encoding=encoding, errors="replace") as f:
        lines = f.readlines()
    if len(lines) < 2:
        raise ValueError("CSV no tiene suficientes líneas para unir encabezado.")
    header = (lines[0].rstrip("\n") + lines[1]).replace("\r", "")
    fixed_lines = [header] + lines[2:]
    df = pd.read_csv(StringIO("".join(fixed_lines)), encoding=encoding, index_col=0)
    if df.index.name is None or str(df.index.name).lower() != ID_COL.lower():
        df.index.name = ID_COL
    return df

df = read_csv_with_header_fix(BASE_CSV_PATH)
log(f"Base leída: shape={df.shape}")

# ---------- 2) Helpers de columnas del CSV ----------
def parse_col(col: str):
    parts = str(col).split("__")
    if len(parts) >= 3:
        return parts[0], parts[1], "__".join(parts[2:])
    return None, None, col

pat_schema = re.compile(r"^[^_]+__[^_]+__.+$")
schema_cols = [c for c in df.columns if isinstance(c, str) and pat_schema.match(c)]
log(f"Columnas con esquema task__family__feature: {len(schema_cols)} / {df.shape[1]}")

# Índices de columnas por tarea
from collections import defaultdict
task_index = {}
all_tasks = sorted({parse_col(c)[0] for c in df.columns if parse_col(c)[0] is not None})
for t in all_tasks:
    famfeat_to_col = {}
    feat_to_cols   = defaultdict(list)
    for c in df.columns:
        task,fam,feat = parse_col(c)
        if task != t:
            continue
        famfeat = f"{fam}__{feat}" if fam and feat else None
        if famfeat:
            famfeat_to_col[famfeat] = c
        if feat:
            feat_to_cols[feat].append(c)
    task_index[t] = dict(famfeat=famfeat_to_col, feat=feat_to_cols)

# ---------- 3) Cargar mapping tarea–modalidad–feature→macrodomain ----------
if not os.path.exists(MACRO_MAP_PATH):
    raise FileNotFoundError(f"No se encuentra el mapping: {MACRO_MAP_PATH}")

map_df = pd.read_csv(MACRO_MAP_PATH)
for col in ["task","modality","feature","macro_domain"]:
    if col not in map_df.columns:
        raise ValueError(f"Falta columna '{col}' en mapping.")
map_df["task"] = map_df["task"].astype(str)
map_df["feature"] = map_df["feature"].astype(str)

# ---------- 4) Construir registro por macrodomain (y opcionalmente por modalidad) ----------
def columns_for_task_feature(task, feature):
    cols = []
    feat_map = task_index.get(task, {}).get("feat", {})
    famfeat_map = task_index.get(task, {}).get("famfeat", {})
    # match exacto de feature
    if feature in feat_map:
        cols.extend(feat_map[feature])
    # si viene family__feature
    if "__" in feature:
        ff = feature
        if ff in famfeat_map:
            cols.append(famfeat_map[ff])
        # tolerar task__fam__feat completo
        if feature.startswith(task + "__"):
            rest = feature[len(task)+2:]
            if rest in famfeat_map:
                cols.append(famfeat_map[rest])
    # únicos
    return list(dict.fromkeys(cols))

registry = {}  # macro_domain -> (optional modality -> set(columns))
kept_pairs = 0
for _, r in map_df.iterrows():
    task = r["task"]
    feature = r["feature"]
    macro = r["macro_domain"]
    modality = r.get("modality", "unspecified")
    cols = columns_for_task_feature(task, feature)
    if not cols:
        continue
    if GROUP_BY_MODALITY:
        registry.setdefault(macro, {}).setdefault(modality, set()).update(cols)
    else:
        registry.setdefault(macro, set()).update(cols)
    kept_pairs += 1

if GROUP_BY_MODALITY:
    total_cols = sum(len(registry[m][mod]) for m in registry for mod in registry[m])
else:
    total_cols = sum(len(registry[m]) for m in registry)

log(f"Mapping a columnas: pares aceptados={kept_pairs} | columnas únicas mapeadas={total_cols}")

# ---------- 5) Preprocesamiento ----------
class CorrDropper:
    def __init__(self, threshold=0.98):
        self.threshold = float(threshold)
        self.keep_idx_ = None
    def fit(self, X):
        X = np.asarray(X, dtype=float)
        var = X.var(axis=0)
        nonconst = np.where(var > 0)[0]
        if len(nonconst) == 0:
            self.keep_idx_ = []
            return self
        Xnc = X[:, nonconst]
        with np.errstate(invalid="ignore"):
            corr = np.corrcoef(Xnc, rowvar=False)
        corr = np.abs(corr)
        np.fill_diagonal(corr, 0.0)
        keep = []
        for j in range(corr.shape[1]):
            if not keep:
                keep.append(j)
                continue
            if np.any(corr[j, keep] > self.threshold):
                continue
            keep.append(j)
        self.keep_idx_ = [int(nonconst[j]) for j in keep]
        return self
    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return X[:, self.keep_idx_] if self.keep_idx_ else X[:, []]
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

def preprocess_matrix(X_df):
    X = X_df.copy().replace([np.inf, -np.inf], np.nan)

    # Imputación
    imp = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imp.fit_transform(X), columns=X.columns, index=X.index)

    # Winsorización
    if WINSORIZE:
        lo, hi = CLIP_PCTS
        for c in X_imp.columns:
            v = X_imp[c].values
            if np.isnan(v).all() or np.nanmin(v) == np.nanmax(v):
                continue
            ql, qh = np.nanpercentile(v, [lo, hi])
            if ql < qh:
                X_imp[c] = np.clip(v, ql, qh)

    # Transformación
    if TRANSFORM_METHOD.lower() == "rankgauss":
        qt = QuantileTransformer(
            output_distribution="normal",
            n_quantiles=min(1000, max(10, len(X_imp))),
            subsample=int(1e9),
            random_state=RANDOM_STATE,
        )
        X_tr = pd.DataFrame(qt.fit_transform(X_imp), columns=X_imp.columns, index=X_imp.index)
    else:
        X_tr = pd.DataFrame(index=X_imp.index)
        for c in X_imp.columns:
            col2d = X_imp[[c]].values
            try:
                pt = PowerTransformer(method="yeo-johnson", standardize=False)
                X_tr[c] = pt.fit_transform(col2d).ravel()
            except Exception:
                qt = QuantileTransformer(
                    output_distribution="normal",
                    n_quantiles=min(1000, max(10, len(X_imp))),
                    subsample=int(1e9),
                    random_state=RANDOM_STATE,
                )
                X_tr[c] = qt.fit_transform(col2d).ravel()

    # Escalado
    scaler = RobustScaler() if ROBUST_SCALE else StandardScaler()
    X_std_full = pd.DataFrame(scaler.fit_transform(X_tr), columns=X_tr.columns, index=X_tr.index)

    # Filtro de redundancia
    if APPLY_REDUNDANCY_FILTER_EXPORT and X_std_full.shape[1] > 1:
        cd = CorrDropper(threshold=REDUNDANCY_THRESHOLD)
        X_mat = cd.fit_transform(X_std_full.values)
        keep_idx = cd.keep_idx_ or []
        keep_names = [X_std_full.columns[i] for i in keep_idx]
        X_std = pd.DataFrame(X_mat, index=X_std_full.index, columns=keep_names)
    else:
        X_std = X_std_full

    return X_std

# ---------- 6) PCA runner ----------
def run_pca_group(cols, group_name):
    use = [c for c in cols
           if c in df.columns
           and np.issubdtype(df[c].dtype, np.number)
           and df[c].notna().sum() > 0
           and df[c].nunique(dropna=True) > 1]

    # Si sólo hay 1 columna utilizable, permitimos k=1
    if len(use) == 1:
        X = df[use].copy()
        Xs = preprocess_matrix(X)
        k = 1
        scores = Xs.values  # una sola componente = la feature escalada
        scores_df = pd.DataFrame(scores, index=X.index, columns=["PC1"])
        scores_df.insert(0, ID_COL, scores_df.index.astype(str))
        var_df = pd.DataFrame({"component": ["PC1"], "explained_variance_ratio": [1.0], "cumulative_variance_ratio": [1.0]})
        loadings_df = pd.DataFrame({"PC1": [1.0]}, index=Xs.columns)  # carga unitaria sobre la única variable
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", group_name)[:120]
        base = os.path.join(OUT_DIR, safe)
        scores_df.to_csv(base + "_scores.csv", index=False)
        loadings_df.to_csv(base + "_loadings.csv", index=True)
        var_df.to_csv(base + "_variance.csv", index=False)

        plt.figure(figsize=(6, 4))
        plt.plot([1], [1.0], marker="o")
        plt.xlabel("Principal Component")
        plt.ylabel("Cumulative Explained Variance")
        plt.title(f"PCA Variance — {group_name}")
        plt.ylim(0, 1.01)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(base + "_variance_plot.png", dpi=150)
        plt.close()

        log(f"{group_name}: k=1 (sólo 1 feature útil).")
        return {"group": group_name, "scores_path": base + "_scores.csv", "n_cols": 1, "k": 1, "var_at_k": 1.0}

    # Si menos que MIN_COMPONENTS_EXPORT y más de 1, se salta
    if len(use) < MIN_COMPONENTS_EXPORT:
        log(f"SKIP {group_name}: columnas utilizables={len(use)} (<{MIN_COMPONENTS_EXPORT})")
        return None

    X = df[use].copy()
    t0 = time.perf_counter()
    Xs = preprocess_matrix(X)
    t1 = time.perf_counter()

    pmax = min(Xs.shape[1], MAX_COMPONENTS_EXPORT)
    pca_full = PCA(n_components=pmax, random_state=RANDOM_STATE).fit(Xs)
    cum = np.cumsum(pca_full.explained_variance_ratio_)
    k = int(np.searchsorted(cum, VAR_TARGET_EXPORT) + 1)
    k = max(MIN_COMPONENTS_EXPORT, min(k, pmax))

    pca = PCA(n_components=k, random_state=RANDOM_STATE).fit(Xs)
    scores = pca.transform(Xs)
    t2 = time.perf_counter()

    scores_df = pd.DataFrame(scores, index=X.index, columns=[f"PC{i+1}" for i in range(k)])
    scores_df.insert(0, ID_COL, scores_df.index.astype(str))

    var_df = pd.DataFrame({
        "component": [f"PC{i+1}" for i in range(k)],
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_variance_ratio": np.cumsum(pca.explained_variance_ratio_),
    })
    loadings_df = pd.DataFrame(pca.components_.T, index=Xs.columns, columns=[f"PC{i+1}" for i in range(k)])

    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", group_name)[:120]
    base = os.path.join(OUT_DIR, safe)
    scores_df.to_csv(base + "_scores.csv", index=False)
    loadings_df.to_csv(base + "_loadings.csv", index=True)
    var_df.to_csv(base + "_variance.csv", index=False)

    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(1, k + 1), var_df["cumulative_variance_ratio"], marker="o")
    plt.xlabel("Principal Component")
    plt.ylabel("Cumulative Explained Variance")
    plt.title(f"PCA Variance — {group_name}")
    plt.ylim(0, 1.01)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(base + "_variance_plot.png", dpi=150)
    plt.close()

    log(f"{group_name}: prep={t1-t0:.2f}s | pca={t2-t1:.2f}s | total={t2-t0:.2f}s | k={k} | cols={len(use)}")
    return {
        "group": group_name,
        "scores_path": base + "_scores.csv",
        "n_cols": len(use),
        "k": k,
        "var_at_k": float(var_df['cumulative_variance_ratio'].iloc[-1]),
    }

# ---------- 7) Ejecutar PCA por macrodomain (y opcionalmente modalidad) ----------
results = []
if GROUP_BY_MODALITY:
    for macro in sorted(registry.keys()):
        for modality in sorted(registry[macro].keys()):
            cols = sorted(list(registry[macro][modality]))
            name = f"macro={macro} | modality={modality}"
            res = run_pca_group(cols, name)
            if res: results.append(res)
else:
    for macro in sorted(registry.keys()):
        cols = sorted(list(registry[macro]))
        name = f"macro={macro}"
        res = run_pca_group(cols, name)
        if res: results.append(res)

summary_df = pd.DataFrame(results)
summary_path = os.path.join(OUT_DIR, "PCA_macros_summary.csv")
summary_df.to_csv(summary_path, index=False)
log(f"Resumen escrito: {summary_path}")

# ---------- 8) Unificar PCs en una sola tabla ----------
merged = None
for rec in results:
    pcs = pd.read_csv(rec["scores_path"])
    if ID_COL not in pcs.columns:
        raise ValueError(f"'{ID_COL}' no está en {rec['scores_path']}")
    pcs = pcs.set_index(ID_COL)
    prefix = re.sub(r"[^A-Za-z0-9]+", "_", rec["group"]).strip("_") + "__"
    pcs = pcs.add_prefix(prefix)
    if merged is None:
        merged = pcs.copy()
    else:
        merged = merged.join(pcs, how="outer")

merged = merged.copy()
merged.insert(0, ID_COL, merged.index.astype(str))
merged_path = os.path.join(OUT_DIR, MERGED_PC_FILENAME)
merged.to_csv(merged_path, index=False)
log(f"Tabla unificada: {merged_path}")

print("\nHecho.")
print("OUT_DIR:", OUT_DIR)
print("Log:", LOG_PATH)