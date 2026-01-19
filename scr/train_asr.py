#Imports e metricas

import os
import sys
import json
import time
import math
import argparse
import inspect
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
if os.name == "nt":
    env_root = Path(sys.executable).resolve().parent
    dll_dir = env_root / "Library" / "bin"
    if dll_dir.exists():
        os.add_dll_directory(str(dll_dir))
        os.environ["PATH"] = str(dll_dir) + os.pathsep + os.environ.get("PATH", "")

PROJECT_ROOT_DEFAULT = Path(__file__).resolve().parents[1]
HF_HOME = PROJECT_ROOT_DEFAULT / ".hf"
HF_HOME.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(HF_HOME)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["DATASETS_DISABLE_MULTIPROCESSING"] = "1"

import numpy as np
import pandas as pd
import torch

from datasets import Dataset, DatasetDict
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
import evaluate

import soundfile as sf
from scipy.signal import resample_poly

wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")
processor: Optional[Wav2Vec2Processor] = None

# -----------------------------
# Vocabulario
# -----------------------------
_ALLOWED_ACCENTS = "áàâãäéèêëíìîïóòôõöúùûüç"
_BASE_CHARS = "abcdefghijklmnopqrstuvwxyz" + _ALLOWED_ACCENTS

def normalize_text_ptbr(
    text: str,
    allow_numbers: bool = True,
    allow_apostrophe: bool = True,
) -> str:
    if text is None:
        return ""
    t = str(text).lower().strip()

    
    t = t.replace("’", "'").replace("`", "'").replace("´", "'")
    t = " ".join(t.split())

    allowed = set(_BASE_CHARS + " ")
    if allow_apostrophe:
        allowed.add("'")
    if allow_numbers:
        allowed.update(list("0123456789"))

    out = []
    for ch in t:
        if ch in allowed:
            out.append(ch)
        else:
            out.append(" ")

    t2 = "".join(out)
    t2 = " ".join(t2.split())
    return t2

def build_vocab_from_texts(
    texts: List[str],
    allow_numbers: bool,
    allow_apostrophe: bool,
) -> Dict[str, int]:
    charset = set()
    for t in texts:
        for ch in t:
            charset.add(ch)

    
    charset.add(" ")  # word delimiter
    if allow_apostrophe:
        charset.add("'")
    if allow_numbers:
        for d in "0123456789":
            charset.add(d)

    allowed = set(_BASE_CHARS + " ")
    if allow_apostrophe:
        allowed.add("'")
    if allow_numbers:
        allowed.update(list("0123456789"))

    charset = sorted([c for c in charset if c in allowed])

    # '|' como separador de palavra
    vocab = {c: i for i, c in enumerate(charset)}
    vocab["|"] = len(vocab)  
    vocab["[UNK]"] = len(vocab)
    vocab["[PAD]"] = len(vocab)
    return vocab


# -----------------------------
# Carregamento dos audios
# -----------------------------
TARGET_SR = 16_000

def load_resample_mono(path: str, target_sr: int = TARGET_SR) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float32", always_2d=True)  
    if wav.size == 0:
        raise RuntimeError(f"Empty audio file: {path}")

    # mono
    wav = wav.mean(axis=1)

    if sr != target_sr:
        # resample_poly: mais estável/rápido que FFT resample
        g = math.gcd(sr, target_sr)
        up = target_sr // g
        down = sr // g
        wav = resample_poly(wav, up, down).astype(np.float32, copy=False)

    # remove NaN/infs
    if not np.isfinite(wav).all():
        wav = np.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    return wav


# -----------------------------
# Fail-fast checks
# -----------------------------
def fail_fast_dependency_checks():
    try:
        import accelerate  # noqa
    except Exception as e:
        raise ImportError(
            "Missing dependency: accelerate. "
            "Install with: python -m pip install -U accelerate"
        ) from e

    # quick audio sanity: soundfile OK
    try:
        _ = sf.available_formats()
    except Exception as e:
        raise RuntimeError(
            "soundfile (libsndfile) não está funcional. "
            "Reinstale com: python -m pip install -U soundfile"
        ) from e


# -----------------------------
# Collator on-the-fly (no ds.map)
# -----------------------------
@dataclass
class DataCollatorCTCOnTheFly:
    processor: Wav2Vec2Processor
    target_sr: int = TARGET_SR
    padding: Union[bool, str] = True
    max_audio_seconds: float = 10.0
    min_audio_seconds: float = 0.2

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = int(self.max_audio_seconds * self.target_sr)
        min_len = int(self.min_audio_seconds * self.target_sr)

        audio_arrays: List[np.ndarray] = []
        texts: List[str] = []

        for f in features:
            wav = load_resample_mono(f["audio"], target_sr=self.target_sr)

            n = len(wav)
            if n < min_len:
                continue
            if n > max_len:
                wav = wav[:max_len]  # truncagem simples (reduz OOM)

            audio_arrays.append(wav)
            texts.append(f["text"])

        # fallback: nunca devolve batch vazio
        if len(audio_arrays) == 0:
            wav = np.zeros(int(1.0 * self.target_sr), dtype=np.float32)
            audio_arrays = [wav]
            texts = [features[0]["text"]]

        batch = self.processor(
            audio_arrays,
            sampling_rate=self.target_sr,
            padding=self.padding,
            return_tensors="pt",
        )

        labels = self.processor(
            text=texts,
            padding=self.padding,
            return_tensors="pt",
        ).input_ids

        labels = labels.masked_fill(labels == self.processor.tokenizer.pad_token_id, -100)
        batch["labels"] = labels
        return batch


def compute_metrics(pred):
    logits = pred.predictions
    pred_ids = np.argmax(logits, axis=-1)
    pred_str = processor.batch_decode(pred_ids)

    label_ids = pred.label_ids.copy()
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, group_tokens=False)

    return {
        "wer": wer_metric.compute(predictions=pred_str, references=label_str),
        "cer": cer_metric.compute(predictions=pred_str, references=label_str),
    }


# -----------------------------
# Args
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--project_root", type=str, default=str(PROJECT_ROOT_DEFAULT))
    p.add_argument("--run_dir", type=str, required=True)

    # ASR-ready (CTC fine-tuned)- Edresson/wav2vec2-large-xlsr-coraa-portuguese. 
    p.add_argument("--base_model", type=str, default="Edresson/wav2vec2-large-xlsr-coraa-portuguese")

    p.add_argument("--hf_token", type=str, default="", help="token HF opcional")

    # texto/vocab
    p.add_argument("--no_numbers", action="store_true", default=False)
    p.add_argument("--no_apostrophe", action="store_true", default=False)

    # áudio/memória
    p.add_argument("--max_audio_seconds", type=float, default=10.0)
    p.add_argument("--min_audio_seconds", type=float, default=0.2)

    # debug/subsample
    p.add_argument("--max_train_samples", type=int, default=0)
    p.add_argument("--max_valid_samples", type=int, default=0)
    p.add_argument("--max_test_samples", type=int, default=0)

    # treino
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_train_epochs", type=int, default=5)
    p.add_argument("--warmup_steps", type=int, default=500)

    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--logging_steps", type=int, default=50)

    p.add_argument("--freeze_feature_encoder", action="store_true", default=True)

    # GPU
    p.add_argument("--use_cuda", action="store_true", default=True)
    p.add_argument("--fp16", action="store_true", default=True)
    p.add_argument("--num_workers", type=int, default=2)

    return p.parse_args()


def main():
    global processor
    args = parse_args()
    fail_fast_dependency_checks()

    run_dir = Path(args.run_dir).resolve()
    splits_dir = run_dir / "splits"
    vocab_dir = run_dir / "vocab"
    ckpt_dir = run_dir / "checkpoints"
    final_dir = run_dir / "final_model"

    for d in [vocab_dir, ckpt_dir, final_dir]:
        d.mkdir(parents=True, exist_ok=True)

    allow_numbers = not args.no_numbers
    allow_apostrophe = not args.no_apostrophe

    # device
    cuda_ok = torch.cuda.is_available() and args.use_cuda
    device = torch.device("cuda" if cuda_ok else "cpu")
    print(f"[train] device: {device} (cuda_available={torch.cuda.is_available()})")

    # load splits
    train_df = pd.read_csv(splits_dir / "train.csv")
    valid_df = pd.read_csv(splits_dir / "valid.csv")
    test_df  = pd.read_csv(splits_dir / "test.csv")

    # optional subsample
    if args.max_train_samples and args.max_train_samples > 0:
        train_df = train_df.sample(args.max_train_samples, random_state=42)
    if args.max_valid_samples and args.max_valid_samples > 0:
        valid_df = valid_df.sample(args.max_valid_samples, random_state=42)
    if args.max_test_samples and args.max_test_samples > 0:
        test_df = test_df.sample(args.max_test_samples, random_state=42)

    # normalize
    train_df["text_norm"] = train_df["text"].apply(lambda t: normalize_text_ptbr(t, allow_numbers, allow_apostrophe))
    valid_df["text_norm"] = valid_df["text"].apply(lambda t: normalize_text_ptbr(t, allow_numbers, allow_apostrophe))
    test_df["text_norm"]  = test_df["text"].apply(lambda t: normalize_text_ptbr(t, allow_numbers, allow_apostrophe))

    # vocab from TRAIN only
    vocab = build_vocab_from_texts(train_df["text_norm"].tolist(), allow_numbers, allow_apostrophe)
    vocab_file = vocab_dir / "vocab.json"
    with open(vocab_file, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"[train] vocab_size={len(vocab)} saved to {vocab_file}")

    tokenizer = Wav2Vec2CTCTokenizer(
        str(vocab_file),
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
    )
    token = args.hf_token.strip() or None

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        args.base_model,
        feature_size=1,
        sampling_rate=TARGET_SR,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
        token=token,
    )

    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    # modelo
    model = Wav2Vec2ForCTC.from_pretrained(
        args.base_model,
        vocab_size=len(processor.tokenizer),
        pad_token_id=processor.tokenizer.pad_token_id,
        ctc_loss_reduction="mean",
        use_safetensors=True,
        ignore_mismatched_sizes=True,  
        token=token,
    )

    # memory savers
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    if args.freeze_feature_encoder:
        if hasattr(model, "freeze_feature_encoder"):
            model.freeze_feature_encoder()
        else:
            model.freeze_feature_extractor()

    model.to(device)

    # datasets (sem map)
    def to_ds(df: pd.DataFrame) -> Dataset:
        return Dataset.from_pandas(
            df[["abs_path", "text_norm"]].rename(columns={"abs_path": "audio", "text_norm": "text"})
        )

    dsd = DatasetDict({
        "train": to_ds(train_df),
        "validation": to_ds(valid_df),
        "test": to_ds(test_df),
    })

    data_collator = DataCollatorCTCOnTheFly(
        processor=processor,
        target_sr=TARGET_SR,
        padding=True,
        max_audio_seconds=float(args.max_audio_seconds),
        min_audio_seconds=float(args.min_audio_seconds),
    )

    # TrainingArguments 
    ta_params = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
    training_kwargs = dict(
        output_dir=str(ckpt_dir),
        remove_unused_columns=False,
        report_to=[],
        per_device_train_batch_size=int(args.batch_size),
        per_device_eval_batch_size=int(args.batch_size),
        gradient_accumulation_steps=int(args.grad_accum),
        dataloader_num_workers=int(args.num_workers),
        learning_rate=float(args.lr),
        warmup_steps=int(args.warmup_steps),
        num_train_epochs=int(args.num_train_epochs),
        save_steps=int(args.save_steps),
        eval_steps=int(args.eval_steps),
        logging_steps=int(args.logging_steps),
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        fp16=bool(args.fp16 and cuda_ok),
        gradient_checkpointing=True,
        group_by_length=False,
    )

    if "evaluation_strategy" in ta_params:
        training_kwargs["evaluation_strategy"] = "steps"
    elif "eval_strategy" in ta_params:
        training_kwargs["eval_strategy"] = "steps"

    training_args = TrainingArguments(**training_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dsd["train"],
        eval_dataset=dsd["validation"],
        tokenizer=processor.feature_extractor,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    with open(run_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)
    t0 = time.time()
    trainer.train()
    print(f"[train] training finished in {time.time()-t0:.1f}s")

    # Validacao
    val_metrics = trainer.evaluate(dsd["validation"])
    test_metrics = trainer.evaluate(dsd["test"])
    print("[RESULT] Validation:", val_metrics)
    print("[RESULT] Test:", test_metrics)

    with open(run_dir / "metrics_validation.json", "w", encoding="utf-8") as f:
        json.dump(val_metrics, f, ensure_ascii=False, indent=2)
    with open(run_dir / "metrics_test.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, ensure_ascii=False, indent=2)

    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))
    print(f"[train] saved: {final_dir}")


if __name__ == "__main__":
    main()
