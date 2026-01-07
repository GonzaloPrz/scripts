import argparse
from pathlib import Path

import pandas as pd


def resolve_base_dir() -> Path:
    home = Path.home()
    if "Users/gp" in str(home):
        return Path(home, "data", "affective_pitch")
    return Path("D:", "CNC_Audio", "gonza", "data", "affective_pitch")


def normalize_group(value: object) -> str | None:
    if pd.isna(value):
        return None
    value = str(value).strip().upper()
    mapping = {
        "CN": "HC",
        "HC": "HC",
        "AD": "AD",
        "FTD": "FTD",
    }
    return mapping.get(value, value)


def parse_args() -> argparse.Namespace:
    base_dir = resolve_base_dir()
    parser = argparse.ArgumentParser(
        description=(
            "Sample a random balanced subset of audios across groups "
            "(HC, FTD, AD)."
        )
    )
    parser.add_argument(
        "--input",
        default=base_dir / "transcripts_fugu_matched_group.csv",
        type=Path,
        help="Input CSV with group labels.",
    )
    parser.add_argument(
        "--output",
        default=base_dir / "subset_for_benchmarking.csv",
        type=Path,
        help="Output CSV with the sampled subset.",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.10,
        help="Target fraction of total rows to sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--group-col",
        default="group",
        help="Column name with group labels.",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=["HC", "FTD", "AD"],
        help="Groups to include (will be normalized).",
    )
    parser.add_argument(
        "--dedupe-cols",
        default="",
        help="Comma-separated columns to dedupe before sampling.",
    )
    return parser.parse_args()


def balanced_sample(
    data: pd.DataFrame,
    group_col: str,
    groups: list[str],
    fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    if group_col not in data.columns:
        raise ValueError(f"Missing group column: {group_col}")

    data = data.copy()
    data["group_norm"] = data[group_col].map(normalize_group)

    groups = [normalize_group(group) for group in groups]
    groups = [group for group in groups if group]
    groups = list(dict.fromkeys(groups))

    data = data[data["group_norm"].isin(groups)]
    if data.empty:
        raise ValueError("No rows after filtering by group.")

    total_rows = len(data)
    per_group = int(round(total_rows * fraction / len(groups)))
    per_group = max(per_group, 1)

    group_sizes = data["group_norm"].value_counts().to_dict()
    missing_groups = [group for group in groups if group_sizes.get(group, 0) == 0]
    if missing_groups:
        raise ValueError(f"Missing groups in data: {missing_groups}")

    min_group_size = min(group_sizes[group] for group in groups)
    if per_group > min_group_size:
        per_group = min_group_size

    samples = []
    for group in groups:
        group_data = data[data["group_norm"] == group]
        sample = group_data.sample(n=per_group, random_state=seed)
        samples.append(sample)

    subset = pd.concat(samples, ignore_index=True)
    subset = subset.sample(frac=1, random_state=seed).reset_index(drop=True)

    stats = {
        "total_rows": total_rows,
        "groups": groups,
        "per_group": per_group,
        "subset_rows": len(subset),
        "group_sizes": group_sizes,
        "subset_group_sizes": subset["group_norm"].value_counts().to_dict(),
    }
    return subset, stats


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)

    if args.dedupe_cols:
        dedupe_cols = [col.strip() for col in args.dedupe_cols.split(",") if col.strip()]
        if dedupe_cols:
            df = df.drop_duplicates(subset=dedupe_cols)

    subset, stats = balanced_sample(
        data=df,
        group_col=args.group_col,
        groups=args.groups,
        fraction=args.fraction,
        seed=args.seed,
    )

    subset.to_csv(args.output, index=False)

    print(f"Total rows: {stats['total_rows']}")
    print(f"Group sizes: {stats['group_sizes']}")
    print(f"Sample per group: {stats['per_group']}")
    print(f"Subset rows: {stats['subset_rows']}")
    print(f"Subset group sizes: {stats['subset_group_sizes']}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
