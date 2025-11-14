import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# 1. Funciones auxiliares
# -----------------------------
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
filename = "best_best_models_roc_auc_5_folds_StandardScaler_count_mean_ratio_bca_hyp_opt_feature_selection.csv"
filename_shuffle = "best_models_roc_auc_5_folds_StandardScaler_count_mean_ratio_bca_hyp_opt_feature_selection_shuffled.csv"

dev_holdout = pd.read_csv(filename)

dev = dev_holdout[['task'] + [col for col in dev_holdout.columns if col.endswith('_dev')]]
holdout = dev_holdout[['task'] + [col for col in dev_holdout.columns if col.endswith('_holdout')]]

shuffle = pd.read_csv(filename_shuffle).rename(columns=lambda x: x.replace('_dev',''))

# -----------------------------
# 3. Extraer AUC + IC en dev y holdout
#    (mejor modelo por tarea)
# -----------------------------

# Añadir columnas numéricas de AUC+CI en dev
dev.columns = ['task'] + [c.replace('_dev','') for c in dev.columns if c != 'task']
holdout = holdout.rename(columns=lambda x: x.replace('_holdout', '') if x != 'task' else x)

dev = add_ci_columns(dev, 'auc', 'auc')
holdout = add_ci_columns(holdout, 'auc', 'auc')
shuffle = add_ci_columns(shuffle, 'auc_shuffle_dev', 'auc_shuffle')
# -----------------------------
# 5. Construir tabla final por tarea
# -----------------------------
summary = pd.DataFrame(index=dev['task'].unique())

summary['auc_dev_mean'] = dev['auc_mean']
summary['auc_dev_low'] = dev['auc_low']
summary['auc_dev_high'] = dev['auc_high']

summary['auc_holdout_mean'] = holdout['auc_mean']
summary['auc_holdout_low'] = holdout['auc_low']
summary['auc_holdout_high'] = holdout['auc_high']

summary['auc_shuffle_mean'] = shuffle['auc_shuffle_mean']
summary['auc_shuffle_low'] = shuffle['auc_shuffle_low']
summary['auc_shuffle_high'] = shuffle['auc_shuffle_high']

# Opcional: ordenar tareas (para que el radar sea más legible)
summary = summary.sort_index()


# -----------------------------
# 6. Función para radar plot
# -----------------------------
def radar_plot_auc(summary_df, use_ci=True, title="AUC por tarea (dev, holdout, shuffle)"):
    """
    Crea un radar plot con:
      - Ejes = tareas
      - Curvas = dev, holdout, shuffle
      - use_ci=True: se rellena con los intervalos de confianza
    """
    tasks = list(summary_df.index)
    n = len(tasks)

    # Ángulos para cada eje
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    # Cerramos el círculo
    angles += angles[:1]

    def values_for(col):
        vals = summary_df[col].tolist()
        return vals + vals[:1]

    # Curvas principales
    dev_vals = values_for('auc_dev_mean')
    hold_vals = values_for('auc_holdout_mean')
    shuf_vals = values_for('auc_shuffle_mean')

    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)

    # Etiquetas de los ejes
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(tasks, fontsize=9)

    # Rango radial aproximado para AUC
    min_val = np.nanmin([summary_df['auc_dev_mean'],
                         summary_df['auc_holdout_mean'],
                         summary_df['auc_shuffle_mean']].values)
    max_val = np.nanmax([summary_df['auc_dev_mean'],
                         summary_df['auc_holdout_mean'],
                         summary_df['auc_shuffle_mean']].values)

    # Un pequeño margen
    r_min = max(0.0, min_val - 0.05)
    r_max = min(1.0, max_val + 0.05)

    ax.set_ylim(r_min, r_max)
    ax.set_yticklabels([])  # quitar etiquetas radiales para que no ensucie

    # Dibujar curvas
    ax.plot(angles, dev_vals, linewidth=2, linestyle='solid', label='Development (AUC)')
    ax.plot(angles, hold_vals, linewidth=2, linestyle='solid', label='Holdout (AUC)')
    ax.plot(angles, shuf_vals, linewidth=2, linestyle='solid', label='Shuffle (AUC)')

    if use_ci:
        # Dev CI
        dev_low = values_for('auc_dev_low')
        dev_high = values_for('auc_dev_high')
        ax.fill_between(angles, dev_low, dev_high, alpha=0.15)

        # Holdout CI
        hold_low = values_for('auc_holdout_low')
        hold_high = values_for('auc_holdout_high')
        ax.fill_between(angles, hold_low, hold_high, alpha=0.15)

        # Shuffle CI
        shuf_low = values_for('auc_shuffle_low')
        shuf_high = values_for('auc_shuffle_high')
        ax.fill_between(angles, shuf_low, shuf_high, alpha=0.15)

    ax.set_title(title, fontsize=12, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))

    plt.tight_layout()
    plt.show()


# -----------------------------
# 7. Ejecutar las dos variantes
# -----------------------------

# Con intervalos de confianza
radar_plot_auc(summary, use_ci=True,
               title="AUC (best model) por tarea con IC - dev, holdout y shuffle")

# Sin intervalos de confianza
radar_plot_auc(summary, use_ci=False,
               title="AUC (best model) por tarea SIN IC - dev, holdout y shuffle")