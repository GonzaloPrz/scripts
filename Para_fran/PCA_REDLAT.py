# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 22:19:03 2025

@author: Agus
"""
"""
PCA agrupado por (tarea, modalidad) usando:
  - Excel de filtros por tarea (1 hoja por tarea; columna 'dominio' = audio / texto)
  - CSV base con encabezado partido arreglado (línea 1 + línea 2)
Incluye:
  - Matching difuso hoja<->tarea (cuando los nombres no son idénticos)
  - Uso de columna id como índice
  - Export de scores/loadings/var/plot por grupo
  - Tabla unificada de PCs (incluye columna 'id')
  - Reportes de trazabilidad
"""

# ===================== CONFIG =====================
IN_PATH  = "D:/LAB/Becarios/REDLAT/REDLAT_features_subset.csv"                 # CSV original (encabezado partido)
OUT_DIR  = "D:/LAB/Becarios/REDLAT/data/pca_outputs_redlat"                   # carpeta de salida
TASK_FILTER_XLSX = "D:/LAB/Becarios/REDLAT/REDLAT_features_por_tarea_ok4.xlsx"  # Excel: 1 hoja por tarea con 'dominio'

ID_COL = "id"  # nombre de la columna ID (en el CSV aparece como 'id')
# PCA
VAR_TARGET_EXPORT       = 0.85
MIN_COMPONENTS_EXPORT   = 2
MAX_COMPONENTS_EXPORT   = 30
# Preprocesado
TRANSFORM_METHOD        = "rankgauss"  # "rankgauss" | "yeojohnson"
ROBUST_SCALE            = False        # False=StandardScaler | True=RobustScaler
WINSORIZE               = True
CLIP_PCTS               = (0.5, 99.5)
# Redundancia
APPLY_REDUNDANCY_FILTER_EXPORT = True
REDUNDANCY_THRESHOLD           = 0.98
# Reportes
WRITE_DIAGNOSTIC_REPORTS = True
ACCEPTED_CSV_NAME        = "accepted_features.csv"
DROPPED_CSV_NAME         = "dropped_features.csv"
UNMATCHED_CSV_NAME       = "unmatched_patterns.csv"
MERGED_PC_FILENAME       = "ALL_GROUPS_PC_scores.csv"

RANDOM_STATE = 42
# ===================================

import os, re, csv, time
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import QuantileTransformer, PowerTransformer, StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin

try:
    from tqdm.auto import tqdm
    HAVE_TQDM = True
except Exception:
    HAVE_TQDM = False

# ---------- util/log ----------
def ts(): return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
def fmt_secs(s): m, sec = divmod(int(s), 60); h, m = divmod(m, 60); return f"{h:d}:{m:02d}:{sec:02d}"
def log(msg, logfile):
    line = f"[{ts()}] {msg}"; print(line, flush=True); open(logfile,"a",encoding="utf-8").write(line+"\n")

os.makedirs(OUT_DIR, exist_ok=True)
LOG_PATH = os.path.join(OUT_DIR, "run_log.txt")
open(LOG_PATH, "w", encoding="utf-8").write(f"[{ts()}] Run started. OUT_DIR={OUT_DIR}\n")

# ---------- LECTURA CSV (arreglo encabezado partido) ----------
from io import StringIO

def read_csv_with_header_fix(path, encoding="utf-8"):
    """
    Une las dos primeras líneas (header partido) y lee el CSV resultante.
    Asume separador coma.
    """
    with open(path, "r", encoding=encoding, errors="replace") as f:
        lines = f.readlines()
    if len(lines) < 2:
        raise ValueError("El CSV no tiene suficientes líneas para unir encabezado (se esperaban al menos 2).")
    header = (lines[0].rstrip("\n") + lines[1]).replace("\r", "")
    fixed_lines = [header] + lines[2:]
    df = pd.read_csv(StringIO("".join(fixed_lines)), encoding=encoding, index_col=0)
    # asegurar nombre del índice
    if df.index.name is None or str(df.index.name).lower() != ID_COL.lower():
        df.index.name = ID_COL
    return df

df = read_csv_with_header_fix(IN_PATH)
log(f"Base OK: shape={df.shape}", LOG_PATH)

# ---------- Coerción numérica básica ----------
def coerce_numeric_like(df_in, exclude_cols=None, min_ratio=0.6):
    if exclude_cols is None: exclude_cols=[]
    X=df_in.copy()
    for c in X.columns:
        if c in exclude_cols: continue
        if np.issubdtype(X[c].dtype, np.number): continue
        s=X[c].astype(str).str.strip()
        s=s.str.replace(r"\s+","",regex=True)
        s=s.str.replace(r"\.(?=\d{3}(\D|$))","",regex=True)  # miles
        s=np.where(~pd.Series(s).str.contains(r"\.",regex=True), pd.Series(s).str.replace(",",".",regex=False), s)
        s=pd.Series(s, index=X.index)
        num=pd.to_numeric(s, errors="coerce")
        if 1.0-num.isna().mean() >= min_ratio:
            X[c]=num
    return X

before_num = sum(np.issubdtype(df[c].dtype, np.number) for c in df.columns)
df = coerce_numeric_like(df, exclude_cols=[])
after_num  = sum(np.issubdtype(df[c].dtype, np.number) for c in df.columns)
log(f"Coerción numérica: antes={before_num}, después={after_num}", LOG_PATH)

# ---------- parsing de nombres ----------
def parse_col(col: str):
    parts = str(col).split("__")
    if len(parts)>=3: return parts[0], parts[1], "__".join(parts[2:])
    return None, None, col

# columnas con esquema esperado
pat_schema = re.compile(r"^[^_]+__[^_]+__.+$")
schema_cols = [c for c in df.columns if isinstance(c,str) and pat_schema.match(c)]
log(f"Preflight: columnas esquema task__family__feature = {len(schema_cols)} de {df.shape[1]}", LOG_PATH)

# ---------- índice por task ----------
from collections import defaultdict
task_index = {}
all_tasks = sorted({parse_col(c)[0] for c in df.columns if parse_col(c)[0] is not None})
for t in all_tasks:
    famfeat_to_col = {}
    feat_to_cols   = defaultdict(list)
    for c in df.columns:
        task,fam,feat = parse_col(c)
        if task!=t: continue
        famfeat = f"{fam}__{feat}" if fam is not None and feat is not None else None
        if famfeat: famfeat_to_col[famfeat]=c
        if feat: feat_to_cols[feat].append(c)
    task_index[t] = dict(famfeat=famfeat_to_col, feat=feat_to_cols)

log(f"Tareas detectadas en CSV: {len(all_tasks)} -> {all_tasks}", LOG_PATH)

# ---------- leer Excel filtros ----------
def normalize_mod(x):
    if x is None: return None
    v=str(x).strip().lower()
    if v in {"audio"}: return "audio"
    if v in {"texto","textual","verbal","linguistic","ling","linguistica","lingüística"}: return "linguistic"
    return None

def load_filters(xlsx_path):
    if not os.path.exists(xlsx_path):
        raise FileNotFoundError(f"No existe Excel: {xlsx_path}")
    xls=pd.ExcelFile(xlsx_path)
    filters=[]
    for sheet in xls.sheet_names:
        dfsh=pd.read_excel(xls, sheet_name=sheet, dtype=str)
        if dfsh.empty: continue
        # detecta columna dominio
        dom_col=None
        lower_map = {str(c).strip().lower(): c for c in dfsh.columns}
        for cand in ["dominio","modality","modalidad","tipo","type","category","categoría","categoria","clase"]:
            if cand in lower_map:
                dom_col = lower_map[cand]
                break
        pattern_cols = [c for c in dfsh.columns if c != dom_col] if dom_col else list(dfsh.columns)
        for _,row in dfsh.iterrows():
            dom = normalize_mod(row.get(dom_col, None)) if dom_col else None
            for pc in pattern_cols:
                val=row.get(pc, None)
                if val is None or str(val).strip()=="":
                    continue
                filters.append(dict(sheet=sheet, raw=str(val).strip(), domain=dom))
    return filters

filters = load_filters(TASK_FILTER_XLSX)
log(f"Filtros leídos del Excel: {len(filters)} celdas con texto", LOG_PATH)

# ---------- map sheet->task ----------
def best_task_for_sheet(sheet_name):
    sn = str(sheet_name).strip().lower()
    for t in all_tasks:
        if sn == t.lower() or sn.startswith(t.lower()):
            return t
    return None

sheet_default_task = {s:best_task_for_sheet(s) for s in pd.ExcelFile(TASK_FILTER_XLSX).sheet_names}

# ---------- parsear filtros ----------
parsed_filters=[]
for f in filters:
    raw = f["raw"]
    domain = f["domain"]
    parts=[p.strip() for p in raw.split(",") if p.strip()!=""]
    task_hint=None; payload=None; domain_hint=None
    if len(parts)>=3:
        task_hint = parts[0]
        domain_hint = normalize_mod(parts[1])
        payload = ",".join(parts[2:])
    elif len(parts)==2:
        maybe_dom = normalize_mod(parts[0])
        if maybe_dom is not None:
            domain_hint = maybe_dom; payload = parts[1]
        else:
            task_hint = parts[0]; payload = parts[1]
    else:
        payload = parts[0]

    task_used = None
    if task_hint and task_hint in all_tasks:
        task_used = task_hint
    elif sheet_default_task.get(f["sheet"]):
        task_used = sheet_default_task[f["sheet"]]
    else:
        if "__" in payload:
            maybe_task = payload.split("__",1)[0]
            if maybe_task in all_tasks:
                task_used = maybe_task

    domain_used = domain or domain_hint

    parsed_filters.append(dict(
        sheet=f["sheet"],
        task=task_used,
        domain=domain_used,
        payload=payload
    ))

# ---------- construir registro (task -> modality -> cols) ----------
registry={}
accepted_rows=[]; dropped_rows=[]; unmatched_rows=[]

for pf in parsed_filters:
    task = pf["task"]
    domain = pf["domain"]
    text  = pf["payload"]
    if not domain or domain not in {"audio","linguistic"}:
        dropped_rows.append({"task":task, "modality":None, "pattern":text, "reason":"dominio_no_definido"})
        continue
    if task is None:
        dropped_rows.append({"task":None, "modality":domain, "pattern":text, "reason":"task_no_definido"})
        continue
    if task not in task_index:
        dropped_rows.append({"task":task, "modality":domain, "pattern":text, "reason":"task_no_en_csv"})
        continue

    famfeat_map = task_index[task]["famfeat"]
    feat_map    = task_index[task]["feat"]

    matched_cols=[]

    if "__" in text:
        ff = text.strip()
        if ff in famfeat_map:
            matched_cols = [famfeat_map[ff]]
        else:
            if text.startswith(task+"__") and text[len(task)+2:] in famfeat_map:
                matched_cols = [famfeat_map[text[len(task)+2:]]]
    else:
        feat = text.strip()
        if feat in feat_map:
            matched_cols = feat_map[feat]

    if matched_cols:
        registry.setdefault(task, {}).setdefault(domain, []).extend(matched_cols)
        for c in matched_cols:
            accepted_rows.append({"task":task, "modality":domain, "column":c, "matched_pattern":text, "sheet":pf["sheet"]})
    else:
        unmatched_rows.append({"sheet":pf["sheet"], "task":task, "modality":domain, "pattern":text})

# limpiar duplicados
for t in list(registry.keys()):
    for m in list(registry[t].keys()):
        registry[t][m] = sorted(list(dict.fromkeys(registry[t][m])))

# reportes
if WRITE_DIAGNOSTIC_REPORTS:
    pd.DataFrame(accepted_rows).to_csv(os.path.join(OUT_DIR, ACCEPTED_CSV_NAME), index=False)
    pd.DataFrame(dropped_rows ).to_csv(os.path.join(OUT_DIR, DROPPED_CSV_NAME), index=False)
    pd.DataFrame(unmatched_rows).to_csv(os.path.join(OUT_DIR, UNMATCHED_CSV_NAME), index=False)

kept = sum(len(registry[t][m]) for t in registry for m in registry[t])
log(f"Filtrado: tareas={len(registry)} | columnas aceptadas={kept} | unmatched={len(unmatched_rows)}", LOG_PATH)

# ---------- PCA helpers ----------
class CorrDropper(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.98): self.threshold=float(threshold); self.keep_idx_=None
    def fit(self, X, y=None):
        X=np.asarray(X, dtype=float); var=X.var(axis=0); nonconst=np.where(var>0)[0]
        if len(nonconst)==0: self.keep_idx_=[]; return self
        Xnc=X[:,nonconst]
        with np.errstate(invalid="ignore"): corr=np.corrcoef(Xnc, rowvar=False)
        corr=np.abs(corr); np.fill_diagonal(corr,0.0)
        keep=[]
        for j in range(corr.shape[1]):
            if not keep: keep.append(j); continue
            if np.any(corr[j,keep]>self.threshold): continue
            keep.append(j)
        self.keep_idx_=[int(nonconst[j]) for j in keep]; return self
    def transform(self, X):
        X=np.asarray(X, dtype=float)
        return X[:, self.keep_idx_] if self.keep_idx_ else X[:, []]

def preprocess_matrix(X_df):
    X=X_df.copy().replace([np.inf,-np.inf], np.nan)
    imp=SimpleImputer(strategy="median"); X_imp=pd.DataFrame(imp.fit_transform(X), columns=X.columns, index=X.index)
    if WINSORIZE:
        lo,hi=CLIP_PCTS
        for c in X_imp.columns:
            v=X_imp[c].values
            if np.isnan(v).all() or np.nanmin(v)==np.nanmax(v): continue
            ql,qh=np.nanpercentile(v,[lo,hi])
            if ql<qh: X_imp[c]=np.clip(v,ql,qh)
    if TRANSFORM_METHOD.lower()=="rankgauss":
        qt=QuantileTransformer(output_distribution="normal",
                               n_quantiles=min(1000, max(10,len(X_imp))),
                               subsample=int(1e9), random_state=RANDOM_STATE)
        X_tr=pd.DataFrame(qt.fit_transform(X_imp), columns=X_imp.columns, index=X_imp.index)
    else:
        X_tr=pd.DataFrame(index=X_imp.index)
        for c in X_imp.columns:
            col2d=X_imp[[c]].values
            try:
                pt=PowerTransformer(method="yeo-johnson", standardize=False)
                X_tr[c]=pt.fit_transform(col2d).ravel()
            except Exception:
                qt=QuantileTransformer(output_distribution="normal",
                                       n_quantiles=min(1000, max(10,len(X_imp))),
                                       subsample=int(1e9), random_state=RANDOM_STATE)
                X_tr[c]=qt.fit_transform(col2d).ravel()
    scaler = RobustScaler() if ROBUST_SCALE else StandardScaler()
    X_std_full=pd.DataFrame(scaler.fit_transform(X_tr), columns=X_tr.columns, index=X_tr.index)

    if APPLY_REDUNDANCY_FILTER_EXPORT and X_std_full.shape[1]>1:
        cd=CorrDropper(threshold=REDUNDANCY_THRESHOLD)
        X_mat=cd.fit_transform(X_std_full.values)
        keep_idx=cd.keep_idx_ or []
        keep_names=[X_std_full.columns[i] for i in keep_idx]
        X_std=pd.DataFrame(X_mat, index=X_std_full.index, columns=keep_names)
    else:
        X_std=X_std_full
    return X_std

def run_pca_group(cols, group_name):
    use=[c for c in cols if c in df.columns and np.issubdtype(df[c].dtype, np.number) and df[c].notna().sum()>0 and df[c].nunique(dropna=True)>1]
    if len(use)<MIN_COMPONENTS_EXPORT:
        log(f"SKIP {group_name}: columnas utilizables={len(use)}", LOG_PATH); return None
    X=df[use].copy()
    t0=time.perf_counter(); Xs=preprocess_matrix(X); t1=time.perf_counter()
    pmax=min(Xs.shape[1], MAX_COMPONENTS_EXPORT)
    pca_full=PCA(n_components=pmax, random_state=RANDOM_STATE).fit(Xs)
    cum=np.cumsum(pca_full.explained_variance_ratio_)
    k=int(np.searchsorted(cum, VAR_TARGET_EXPORT)+1); k=max(MIN_COMPONENTS_EXPORT, min(k, pmax))
    pca=PCA(n_components=k, random_state=RANDOM_STATE).fit(Xs)
    scores=pca.transform(Xs); t2=time.perf_counter()

    scores_df=pd.DataFrame(scores, index=X.index, columns=[f"PC{i+1}" for i in range(k)])
    var_df=pd.DataFrame({"component":[f"PC{i+1}" for i in range(k)],
                         "explained_variance_ratio":pca.explained_variance_ratio_,
                         "cumulative_variance_ratio":np.cumsum(pca.explained_variance_ratio_)})
    loadings_df=pd.DataFrame(pca.components_.T, index=Xs.columns, columns=[f"PC{i+1}" for i in range(k)])

    safe=re.sub(r"[^A-Za-z0-9_.-]+","_",group_name)[:120]; base=os.path.join(OUT_DIR,safe)
    scores_df.to_csv(base+"_scores.csv", index=True, index_label=df.index.name)
    loadings_df.to_csv(base+"_loadings.csv", index=True)
    var_df.to_csv(base+"_variance.csv", index=False)

    plt.figure(figsize=(6,4))
    plt.plot(np.arange(1,k+1), var_df["cumulative_variance_ratio"], marker="o")
    plt.xlabel("Principal Component"); plt.ylabel("Cumulative Explained Variance")
    plt.title(f"PCA Variance — {group_name}"); plt.ylim(0,1.01); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(base+"_variance_plot.png", dpi=150); plt.close()

    log(f"{group_name}: prep={fmt_secs(t1-t0)} | pca={fmt_secs(t2-t1)} | total={fmt_secs(t2-t0)} | k={k} | cols={len(use)}", LOG_PATH)
    return {"group":group_name,"n_features":len(use),"n_components":k,
            "var_at_k":float(var_df['cumulative_variance_ratio'].iloc[-1]),
            "scores_path":base+"_scores.csv","loadings_path":base+"_loadings.csv",
            "variance_path":base+"_variance.csv","plot_path":base+"_variance_plot.png"}

# ---------- ejecutar PCA por grupo ----------
summary=[]
for task in sorted(registry.keys()):
    for mod in ("audio","linguistic"):
        cols = registry[task].get(mod, [])
        if not cols: continue
        res = run_pca_group(cols, f"task={task} | type={mod}")
        if res: summary.append(res)

summary_df = pd.DataFrame(summary) if summary else pd.DataFrame(columns=["group","n_features","n_components","var_at_k","scores_path","loadings_path","variance_path","plot_path"])
summary_path=os.path.join(OUT_DIR,"PCA_group_summary.csv")
summary_df.to_csv(summary_path, index=False)
log(f"Resumen escrito: {summary_path}", LOG_PATH)

# ---------- tabla unificada de PCs (con columna 'id') ----------
merged = pd.DataFrame(index=df.index.copy())
for rec in summary:
    pcs=pd.read_csv(rec["scores_path"], index_col=0) if os.path.exists(rec["scores_path"]) else None
    if pcs is None: continue
    gkey=re.sub(r"[^A-Za-z0-9]+","_",rec["group"]).strip("_")
    merged = merged.join(pcs.add_prefix(gkey+"__"), how="left")

# insertar columna 'id' explícita al inicio
merged.insert(0, ID_COL, merged.index.astype(str))

merged_path=os.path.join(OUT_DIR, MERGED_PC_FILENAME)
merged.to_csv(merged_path, index=False)  # ya lleva 'id' como columna, sin índice adicional
log(f"Tabla unificada: {merged_path}", LOG_PATH)

print("\nHecho.")
print("OUT_DIR:", OUT_DIR)
print("Log:", LOG_PATH)
