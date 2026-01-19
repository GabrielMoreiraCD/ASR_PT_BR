#Treinamento de ASR_PT_BR usando Wav2vec, treinado com audios COORA e Reunioes_Luza
import os
import sys
import json
import time
import argparse
import inspect
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Union, Optional

os.environ["DATASETS_DISABLE_MULTIPROCESSING"] = "1"

PROJECT_ROOT_DEFAULT = Path(__file__).resolve().parents[1]
HF_HOME = PROJECT_ROOT_DEFAULT / ".hf"
HF_HOME.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(HF_HOME)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import numpy as np
import pandas as pd
import torch

from datasets import Dataset, DatasetDict
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
import evaluate

from utils_text import build_vocab_from_texts, normalize_text_ptbr
from utils_audio import load_resample_mono, TARGET_SR
from utils_blob import download_blob_to_cache

wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

processor: Optional[Wav2Vec2Processor] = None


def fail_fast_dependency_checks():
    try:
        import accelerate  # noqa
    except Exception as e:
        raise ImportError(
            "Missing dependency: accelerate. Install with:\n"
            "  python -m pip install -U 'accelerate>=0.30'\n"
        ) from e


def is_blob_uri(x: str) -> bool:
    return isinstance(x, str) and x.startswith("blob://")


@dataclass
class DataCollatorCTCOnTheFly:
    processor: Wav2Vec2Processor
    cache_dir: Path
    padding: Union[bool, str] = True
    max_audio_seconds: float = 12.0
    min_audio_seconds: float = 0.2

    # Blob Autorization - tenho que colocar em um .env
    connection_string_env: str = "AZURE_STORAGE_CONNECTION_STRING"

    def _get_local_path(self, uri_or_path: str) -> str:
        if is_blob_uri(uri_or_path):
            conn = os.environ.get(self.connection_string_env, "").strip()
            if not conn:
                raise RuntimeError(
                    f"Missing env var {self.connection_string_env}. "
                    "Set it as a secret variable in Azure DevOps."
                )
            local = download_blob_to_cache(
                blob_uri=uri_or_path,
                cache_dir=self.cache_dir,
                connection_string=conn,
            )
            return str(local)
        return uri_or_path

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = int(self.max_audio_seconds * TARGET_SR)
        min_len = int(self.min_audio_seconds * TARGET_SR)

        audio_arrays: List[np.ndarray] = []
        texts: List[str] = []

        for f in features:
            local_path = self._get_local_path(f["audio"])
            wav = load_resample_mono(local_path, target_sr=TARGET_SR)
            if wav is None:
                continue
            n = len(wav)
            if n < min_len:
                continue
            if n > max_len:
                wav = wav[:max_len]

            audio_arrays.append(wav)
            texts.append(f["text"])

        if not audio_arrays:
            wav = np.zeros(int(1.0 * TARGET_SR), dtype=np.float32)
            audio_arrays = [wav]
            texts = [features[0]["text"]]

        batch = self.processor(
            audio_arrays,
            sampling_rate=TARGET_SR,
            padding=self.padding,
            return_tensors="pt",
        )

        labels = self.processor(text=texts, padding=self.padding, return_tensors="pt").input_ids
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project_root", type=str, default=str(PROJECT_ROOT_DEFAULT))
    p.add_argument("--run_dir", type=str, required=True)

    p.add_argument("--base_model", type=str, default="freds0/distil-whisper-large-v3-ptbr")
    p.add_argument("--max_train_samples", type=int, default=0)
    p.add_argument("--max_valid_samples", type=int, default=0)
    p.add_argument("--max_test_samples", type=int, default=0)

    # texto
    p.add_argument("--allow_numbers", action="store_true", default=True)
    p.add_argument("--no_numbers", action="store_true", default=False)
    p.add_argument("--allow_apostrophe", action="store_true", default=False)
    p.add_argument("--no_apostrophe", action="store_true", default=False)

    # treino CPU 
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_train_epochs", type=int, default=2)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--freeze_feature_encoder", action="store_true", default=True)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--max_audio_seconds", type=float, default=12.0)
    p.add_argument("--min_audio_seconds", type=float, default=0.2)
    p.add_argument("--audio_cache_dir", type=str, default=str(PROJECT_ROOT_DEFAULT / "cache" / "audio"))
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

    allow_numbers = args.allow_numbers and (not args.no_numbers)
    allow_apostrophe = args.allow_apostrophe and (not args.no_apostrophe)

    device = torch.device("cpu")
    print(f"[train] device: {device}")

    train_df = pd.read_csv(splits_dir / "train.csv")
    valid_df = pd.read_csv(splits_dir / "valid.csv")
    test_df  = pd.read_csv(splits_dir / "test.csv")

    if args.max_train_samples and args.max_train_samples > 0:
        train_df = train_df.sample(args.max_train_samples, random_state=42)
    if args.max_valid_samples and args.max_valid_samples > 0:
        valid_df = valid_df.sample(args.max_valid_samples, random_state=42)
    if args.max_test_samples and args.max_test_samples > 0:
        test_df = test_df.sample(args.max_test_samples, random_state=42)

    # normalização para PT de reuniões
    train_df["text_norm"] = train_df["text"].apply(lambda t: normalize_text_ptbr(t, allow_numbers, allow_apostrophe))
    valid_df["text_norm"] = valid_df["text"].apply(lambda t: normalize_text_ptbr(t, allow_numbers, allow_apostrophe))
    test_df["text_norm"]  = test_df["text"].apply(lambda t: normalize_text_ptbr(t, allow_numbers, allow_apostrophe))

    vocab = build_vocab_from_texts(train_df["text_norm"].tolist(), allow_numbers, allow_apostrophe)
    vocab_file = vocab_dir / "vocab.json"
    with open(vocab_file, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    tokenizer = Wav2Vec2CTCTokenizer(
        str(vocab_file),
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
    )

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        args.base_model,
        feature_size=1,
        sampling_rate=TARGET_SR,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )

    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    model = Wav2Vec2ForCTC.from_pretrained(
        args.base_model,
        vocab_size=len(processor.tokenizer),
        pad_token_id=processor.tokenizer.pad_token_id,
        ctc_loss_reduction="mean",
        ignore_mismatched_sizes=True,
    )
    if args.freeze_feature_encoder:
        if hasattr(model, "freeze_feature_encoder"):
            model.freeze_feature_encoder()
        else:
            model.freeze_feature_extractor()

    model.to(device)

    def to_ds(df: pd.DataFrame) -> Dataset:
        return Dataset.from_pandas(df[["abs_path", "text_norm"]].rename(columns={"abs_path": "audio", "text_norm": "text"}))

    dsd = DatasetDict({
        "train": to_ds(train_df),
        "validation": to_ds(valid_df),
        "test": to_ds(test_df),
    })

    audio_cache_dir = Path(args.audio_cache_dir).resolve()
    data_collator = DataCollatorCTCOnTheFly(
        processor=processor,
        cache_dir=audio_cache_dir,
        padding=True,
        max_audio_seconds=args.max_audio_seconds,
        min_audio_seconds=args.min_audio_seconds,
    )

    ta_params = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
    training_kwargs = dict(
        output_dir=str(ckpt_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        dataloader_num_workers=int(args.num_workers),

        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_train_epochs,

        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        save_total_limit=2,

        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,

        report_to=[],
        remove_unused_columns=False,
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
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    with open(run_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    t0 = time.time()
    trainer.train()
    print(f"[train] training finished in {time.time()-t0:.1f}s")

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
