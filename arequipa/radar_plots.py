import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

# -----------------------------
# 1. Funciones auxiliares
# -----------------------------

sns.set_theme(style="whitegrid")  # Fondo blanco con grid sutil

plt.rcParams.update({
    "font.family": "Arial",
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14
})

def parse_ci_cell(x):
    """
    Convierte cadenas del tipo:
        '0.77784, (0.61602, 0.89384)'
    en (mean, low, high).

    Si no hay paréntesis, asume que es solo el valor medio.
    """
    if pd.isna(x):
        return np.nan, np.nan, np.nan

    s = str(x).strip()

    # Intentar patrón "mean, (low, high)"
    m = re.match(r'\s*([0-9\.eE+-]+)\s*,\s*\(([^,]+),\s*([^)]+)\)', s)
    if m:
        mean = float(m.group(1))
        low_str, high_str = m.group(2).strip(), m.group(3).strip()

        low = float(low_str) if low_str.lower() != 'nan' else np.nan
        high = float(high_str) if high_str.lower() != 'nan' else np.nan
        return mean, low, high

    # Si no coincide el patrón, intentar leerlo como un número suelto
    try:
        mean = float(s)
        return mean, np.nan, np.nan
    except ValueError:
        return np.nan, np.nan, np.nan


def add_ci_columns(df, col_name, new_prefix):
    """
    A partir de una columna de texto con 'mean, (low, high)' crea
    tres columnas:
        f'{new_prefix}_mean', f'{new_prefix}_low', f'{new_prefix}_high'
    """
    parsed = df[col_name].apply(lambda v: pd.Series(parse_ci_cell(v)))
    parsed.columns = [f'{new_prefix}_mean', f'{new_prefix}_low', f'{new_prefix}_high']
    return pd.concat([df, parsed], axis=1)


# -----------------------------
# 2. Cargar datos desde CSV
# -----------------------------
base_dir = Path('D:','CNC_Audio','gonza','results','arequipa')
filename = "best_best_models_roc_auc_5_folds_count_mean_ratio_bca_hyp_opt_feature_selection.csv"
filename_shuffle = "best_models_roc_auc_5_folds_StandardScaler_count_mean_ratio_bca_hyp_opt_feature_selection_shuffled.csv"

dev_holdout = pd.read_csv(Path(base_dir,filename))

dev = dev_holdout[['task'] + [col for col in dev_holdout.columns if col.endswith('_dev')]]
holdout = dev_holdout[['task'] + [col for col in dev_holdout.columns if col.endswith('_holdout')]]

shuffle = pd.read_csv(Path(base_dir,filename_shuffle)).rename(columns=lambda x: x.replace('_dev',''))

# -----------------------------
# 3. Extraer AUC + IC en dev y holdout
#    (mejor modelo por tarea)
# -----------------------------

# Añadir columnas numéricas de AUC+CI en dev
dev.columns = ['task'] + [c.replace('_dev','') for c in dev.columns if c != 'task']
holdout = holdout.rename(columns=lambda x: x.replace('_holdout', '') if x != 'task' else x)

dev = add_ci_columns(dev, 'roc_auc', 'auc')
holdout = add_ci_columns(holdout, 'roc_auc', 'auc')
shuffle = add_ci_columns(shuffle, 'roc_auc', 'auc_shuffle')
# -----------------------------
# 5. Construir tabla final por tarea
# -----------------------------
labels_dict = {'craft': 'Story retelling',
               'fugu':'Video narration',
               'lamina2': 'Picture description',
               'dia_tipico': 'Routine description',
               'recuerdo_agradable': 'Memory narration',
               'dia_tipico__recuerdo_agradable': 'All prompt-free \ntasks',
               'craft__fugu__lamina2': 'All prompt-based \ntasks'}

summary = pd.DataFrame(index=dev['task'].unique())

summary['auc_dev_mean'] = dev['auc_mean'].values
summary['auc_dev_low'] = dev['auc_low'].values
summary['auc_dev_high'] = dev['auc_high'].values

summary['auc_holdout_mean'] = holdout['auc_mean'].values
summary['auc_holdout_low'] = holdout['auc_low'].values
summary['auc_holdout_high'] = holdout['auc_high'].values

summary['auc_shuffle_mean'] = shuffle['auc_shuffle_mean'].values
summary['auc_shuffle_low'] = shuffle['auc_shuffle_low'].values
summary['auc_shuffle_high'] = shuffle['auc_shuffle_high'].values

summary.index = summary.index.map(labels_dict)

# Opcional: ordenar tareas (para que el radar sea más legible)
summary = summary.sort_index()

def radar_plot_auc(summary_df, use_ci, path_to_save, title="AUC"):
    summary_df = summary_df.sort_values(by='auc_dev_mean', ascending=False)
    tasks = list(summary_df.index)
    n = len(tasks)

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    def values_for(col):
        vals = summary_df[col].tolist()
        return vals + vals[:1]

    dev_vals  = values_for('auc_dev_mean')
    shuf_vals = values_for('auc_shuffle_mean')

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)

    ax.spines['polar'].set_visible(False)
    # --- rango radial ---
    min_val = np.nanmin(np.concatenate((summary_df['auc_dev_mean'].values,
                                        summary_df['auc_shuffle_mean'].values)))
    
    r_min = np.round(max(0.0, min_val - 0.05), 1)
    r_max = 0.9

    # margen extra para poner los labels
    r_margin = (r_max - r_min) * 0.08   # ajusta 0.08 a gusto
    r_labels = r_max + r_margin

    # ahora el límite llega hasta los labels, no hasta el último círculo
    ax.set_ylim(r_min,r_max)

    # círculos radiales (el último sigue en r_max)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
    ax.set_yticklabels(['0.5', '0.6','0.7','0.8','0.9'],
                       fontsize=12, color='grey')

    # quitamos los xticks por defecto
    ax.set_xticks([])

    # textos en el radio r_labels (un pelín por fuera del círculo)
    for ang, task in zip(angles[:-1], tasks):
        angle_deg = np.degrees(ang)
        rot = angle_deg + 90
        if 90 < rot < 270:
            rot += 180

        ax.text(ang, r_labels, task,
                ha='center', va='center',
                fontsize=14,
                rotation=rot,
                rotation_mode='anchor')

    # resto del plot igual
    ax.yaxis.grid(True)
    ax.plot(angles, dev_vals,  linewidth=2, linestyle='solid', label='Actual labels')
    ax.plot(angles, shuf_vals, linewidth=2, linestyle='solid', label='Shuffled labels')

    if use_ci:
        dev_low  = values_for('auc_dev_low')
        dev_high = values_for('auc_dev_high')
        shuf_low = values_for('auc_shuffle_low')
        shuf_high= values_for('auc_shuffle_high')
        ax.fill_between(angles, dev_low,  dev_high,  alpha=0.15, label='95% CI')
        ax.fill_between(angles, shuf_low, shuf_high, alpha=0.15)

    plt.suptitle(title, fontsize=24, y=1.05)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), fontsize=16)

    plt.tight_layout()
    plt.savefig(path_to_save, dpi=300)
    plt.savefig(path_to_save.with_suffix('.svg'), dpi=300)

# Con intervalos de confianza
radar_plot_auc(summary, use_ci=True,
               path_to_save=Path(base_dir,'radarplot.png'))
