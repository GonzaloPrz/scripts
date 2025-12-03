import re
import pandas as pd
import os

# ================== CONFIG ==================
xlsx_path = r"D:/LAB/Becarios/REDLAT/REDLAT_features_por_tarea_ok4.xlsx"
out_map_path = r"D:/LAB/Becarios/REDLAT/feature_macrodomains_map.csv"
out_summary_path = r"D:/LAB/Becarios/REDLAT/feature_macrodomains_summary.csv"
# ============================================

# Posibles nombres de columna para la modalidad (audio / lingüístico)
MOD_COL_CANDIDATES = {
    "dominio","modality","modalidad","tipo","type","category","categoría","categoria","clase"
}

def normalize_mod(x):
    if pd.isna(x):
        return None
    v = str(x).strip().lower()
    if v in {"audio","voz","oral","speech","acoustic","prosodic"}:
        return "audio"
    if v in {"texto","textual","linguistic","ling","linguistica","lingüística","verbal"}:
        return "linguistic"
    return None

# ====== MACRODOMINIOS (incluye granularity en Memoria_Semantica) ======
MACRO_PATTERNS = {
    "Memoria_Semantica": [
        r"\bconcreteness\b", r"imageabil", r"semantic[_-]?divers", r"hypernym",
        r"\bfreq(uency)?\b|\blog[_-]?frq\b", r"psycholinguistic_objective",
        r"\b(ttr|mattr|hdd|hd[-_]?d)\b", r"type[_-]?token", r"unique_(lemma|words?)",
        r"lexical[_-]?density",
        r"granularity[_-]?extraction"  # <- aquí el cambio
    ],
    "Affect_Emotion": [
        r"sentiment[_-]?analysis", r"\bneg\b|\bneu\b|\bpos\b",
        r"\b(anger|disgust|fear|joy|sadness|surprise)\b", r"valence|arousal|dominance|emotion"
    ],
    "Lexical_Diversity_POSMix": [
        r"metrics[_-]?token", r"metrics[_-]?type", r"content[_-]?words?", r"function[_-]?words?",
        r"\bnoun(s)?\b|\bverb(s)?\b|\badj(ective)?s?\b|\badverb(s)?\b",
        r"pron(_vs_(all|content))?|pronoun", r"proportional[_-]?density",
        r"pron_vs_(all|content)", r"metrics_token_words_quantity",
        r"metrics_type_(words|content_words)_(diversity|quantity)"
    ],
    "Morphosyntax_Complexity": [
        r"\bmlu\b", r"mean[_-]?length", r"\bosv\b|\bsvo\b",
        r"dependency|non[-_]?dependency|core|non[-_]?core",
        r"clause|subordinat|agreement|morpho"
    ],
    "Fluency_Timing": [
        r"talking[_-]?intervals", r"speechrate|articulation[_-]?rate|wpm",
        r"\bpause(s)?\b|mean[_-]?pause|pause[_-]?duration|pause[_-]?variab",
        r"syllable[_-]?duration|npauses"
    ],
    "Prosody_AcousticPhonatory": [
        r"pitch[_-]?analysis|hnr|jitter|shimmer|pitch(_(mean|max|stddev|skewness|kurtosis))?",
        r"intensity|energy|formant|mfcc|spectral|prosod|vowel[_-]?space|rhythm|npvi"
    ],
    "Discourse_Coherence": [
        r"cohesion|connective|discourse|corefer|entity[_-]?grid|topic",
        r"idea[_-]?density|story[_-]?grammar|macrostructure"
    ],
}

def assign_macro(feature_name: str, task: str, modality: str):
    name = feature_name.lower()
    # 1) regex directo por feature
    for macro, pats in MACRO_PATTERNS.items():
        if any(re.search(p, name) for p in pats):
            return macro
    # 2) sesgo por tarea si no matcheó nada
    t = (task or "").lower()
    if t.startswith("fugu"):
        return "Affect_Emotion" if modality != "audio" else "Prosody_AcousticPhonatory"
    if t.startswith("craft") or t.startswith("semantic") or "boat" in t:
        return "Memoria_Semantica"
    if "phonolog" in t:
        return "Prosody_AcousticPhonatory" if modality == "audio" else "Lexical_Diversity_POSMix"
    # 3) fallback conservador
    return "Lexical_Diversity_POSMix"

def build_mapping_from_excel(xlsx_path: str) -> pd.DataFrame:
    xls = pd.ExcelFile(xlsx_path)
    rows = []
    for task in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=task)
        if df is None or df.empty:
            continue
        lower_map = {str(c).strip().lower(): c for c in df.columns}
        mod_col = next((lower_map[c] for c in MOD_COL_CANDIDATES if c in lower_map), None)
        feat_cols = [c for c in df.columns if c != mod_col]
        for _, r in df.iterrows():
            modality = normalize_mod(r[mod_col]) if (mod_col is not None and mod_col in r) else None
            modality = modality or "unspecified"
            for c in feat_cols:
                val = r.get(c, None)
                if pd.isna(val):
                    continue
                feature = str(val).strip()
                if not feature:
                    continue
                rows.append({"task": task, "modality": modality, "feature": feature})
    if not rows:
        return pd.DataFrame(columns=["task","modality","feature","macro_domain"])
    raw_df = pd.DataFrame(rows)
    raw_df["macro_domain"] = raw_df.apply(
        lambda r: assign_macro(r["feature"], r["task"], r["modality"]),
        axis=1
    )
    return raw_df

def main():
    os.makedirs(os.path.dirname(out_map_path), exist_ok=True)
    os.makedirs(os.path.dirname(out_summary_path), exist_ok=True)

    mapping_df = build_mapping_from_excel(xlsx_path)
    mapping_df.to_csv(out_map_path, index=False)

    summary = (
        mapping_df.groupby(["task","modality","macro_domain"])
                  .size()
                  .reset_index(name="n_features")
                  .sort_values(["task","modality","macro_domain"])
    )
    summary.to_csv(out_summary_path, index=False)

    print("OK")
    print("Mapping  ->", out_map_path)
    print("Summary  ->", out_summary_path)
    print("Counts   ->")
    print(mapping_df["macro_domain"].value_counts())

if __name__ == "__main__":
    main()