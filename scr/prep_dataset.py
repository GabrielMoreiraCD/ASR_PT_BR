# src/prep_dataset.py
from __future__ import annotations
import os
import json
import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from utils_text import normalize_text_ptbr
from utils_audio import safe_audio_duration_sec
from utils_onedrive import download_and_extract_zip, transform_onedrive_url

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project_root", type=str, default=str(Path(__file__).resolve().parents[1]))
    # Agora recebemos a URL do OneDrive
    p.add_argument("--onedrive_url", type=str, required=True, help="Link de compartilhamento da pasta contendo 'train' e 'labels'")
    p.add_argument("--run_dir", type=str, default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test_size", type=float, default=0.10)
    p.add_argument("--valid_size", type=float, default=0.10)
    p.add_argument("--min_text_len", type=int, default=1)
    p.add_argument("--max_text_len", type=int, default=300)
    p.add_argument("--allow_numbers", action="store_true", default=True)
    p.add_argument("--no_numbers", action="store_true", default=False)
    p.add_argument("--allow_apostrophe", action="store_true", default=False)
    p.add_argument("--no_apostrophe", action="store_true", default=False)
    p.add_argument("--duration_sample_n", type=int, default=2000)
    return p.parse_args()

def main():
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    
    # Pasta onde os dados brutos ficarão
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download e Extração
    download_url = transform_onedrive_url(args.onedrive_url)
    if not (data_dir / "labels" / "metadata_train_final.csv").exists():
        download_and_extract_zip(download_url, data_dir)
    else:
        print("[prep] Dados parecem já existir, pulando download.")

    # Ajuste dos caminhos baseados na estrutura extraída
    # O Zip do OneDrive geralmente cria a estrutura baseada no nome da pasta ou extrai direto
    # Vamos assumir que metadata_train_final.csv está em data/labels/
    
    csv_path = data_dir / "labels" / "metadata_train_final.csv"
    audio_root = data_dir 
    
    if not csv_path.exists():
        # Fallback: as vezes o zip cria uma pasta raiz com o nome do share
        # Tenta encontrar o csv recursivamente
        found_csvs = list(data_dir.rglob("metadata_train_final.csv"))
        if not found_csvs:
            raise FileNotFoundError("Não encontrei metadata_train_final.csv após extração.")
        csv_path = found_csvs[0]
        # O audio root deve ser o pai da pasta 'labels'
        audio_root = csv_path.parent.parent

    print(f"[prep] csv_path encontrado: {csv_path}")
    print(f"[prep] audio_root definido como: {audio_root}")

    run_dir = Path(args.run_dir) if args.run_dir else (project_root / "outputs" / "asr_runs" / "run_latest")
    run_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = run_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    allow_numbers = args.allow_numbers and (not args.no_numbers)
    allow_apostrophe = args.allow_apostrophe and (not args.no_apostrophe)

    df = pd.read_csv(csv_path)
    if "variety" in df.columns:
        df = df[df["variety"].astype(str).str.lower() == "pt_br"].copy()

    # Ajuste para garantir que file_path não comece com / ou \
    df["file_path"] = df["file_path"].astype(str).apply(lambda x: x.lstrip("/").lstrip("\\"))
    df["abs_path"] = df["file_path"].apply(lambda p: str((audio_root / p).resolve()))
    df["text_norm"] = df["text"].apply(lambda t: normalize_text_ptbr(t, allow_numbers=allow_numbers, allow_apostrophe=allow_apostrophe))

    df = df[df["text_norm"].str.len() >= args.min_text_len].copy()
    df = df[df["text_norm"].str.len() <= args.max_text_len].copy()

    # Validação de existência física
    before = len(df)
    df = df[df["abs_path"].apply(os.path.exists)].copy()
    print(f"[prep] file exists filter: {before} -> {len(df)}")
    
    if len(df) == 0:
        raise RuntimeError("Nenhum arquivo de áudio foi encontrado. Verifique a estrutura da pasta extraída.")

    stratify = df["dataset"] if "dataset" in df.columns else None
    train_df, test_df = train_test_split(df, test_size=args.test_size, random_state=args.seed, stratify=stratify)

    stratify_train = train_df["dataset"] if "dataset" in train_df.columns else None
    train_df, valid_df = train_test_split(
        train_df,
        test_size=args.valid_size / (1.0 - args.test_size),
        random_state=args.seed,
        stratify=stratify_train,
    )

    train_df.to_parquet(splits_dir / "train.parquet", index=False)
    valid_df.to_parquet(splits_dir / "valid.parquet", index=False)
    test_df.to_parquet(splits_dir / "test.parquet", index=False)
    
    stats = {
        "n_total": int(len(df)),
        "n_train": int(len(train_df)),
        "n_valid": int(len(valid_df)),
        "n_test": int(len(test_df)),
        "min_text_len": int(train_df["text_norm"].str.len().min()),
        "max_text_len": int(train_df["text_norm"].str.len().max()),
        "mean_text_len": float(train_df["text_norm"].str.len().mean()),
        "allow_numbers": bool(allow_numbers),
        "allow_apostrophe": bool(allow_apostrophe),
    }

    sample_n = min(args.duration_sample_n, len(train_df))
    if sample_n > 0:
        sample_paths = train_df["abs_path"].sample(sample_n, random_state=args.seed).tolist()
        durs = [safe_audio_duration_sec(p) for p in sample_paths]
        durs = [d for d in durs if d == d]
        if durs:
            durs_sorted = sorted(durs)
            stats.update({
                "duration_sample_n": int(sample_n),
                "duration_mean_sec": float(sum(durs_sorted) / len(durs_sorted)),
                "duration_p95_sec": float(durs_sorted[int(0.95 * (len(durs_sorted) - 1))]),
            })

    with open(run_dir / "prep_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("[prep] saved splits + prep_stats.json")

if __name__ == "__main__":
    main()