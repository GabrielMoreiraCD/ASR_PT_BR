# src/train_asr.py
from __future__ import annotations
import os
import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
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
from dataclasses import dataclass
from typing import Dict, List, Union

from utils_text import build_vocab_from_texts, normalize_text_ptbr
from utils_audio import load_resample_mono, TARGET_SR

# Desativa multiprocessing do datasets para evitar problemas no Azure
os.environ["DATASETS_DISABLE_MULTIPROCESSING"] = "1"

# Métricas
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data Collator padrão para CTC. 
    Assume que inputs já são arrays de audio processados ou caminhos locais.
    Neste script, faremos o processamento do áudio dentro do Dataset map ou on-the-fly simples.
    """
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: int = None
    max_length_labels: int = None
    pad_to_multiple_of: int = None
    pad_to_multiple_of_labels: int = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    return {
        "wer": wer_metric.compute(predictions=pred_str, references=label_str),
        "cer": cer_metric.compute(predictions=pred_str, references=label_str),
    }

def prepare_dataset(batch):
    audio = batch["audio"]
    wav = load_resample_mono(audio, target_sr=TARGET_SR)
    # Em produção ideal, filtramos isso antes. Aqui vamos retornar zero padding se falhar.
    if wav is None:
        wav = np.zeros(16000, dtype=np.float32)

    batch["input_values"] = processor(wav, sampling_rate=TARGET_SR).input_values[0]
    
    with processor.as_target_processor():
        batch["labels"] = processor(batch["text"]).input_ids
        
    return batch

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, required=True)
    p.add_argument("--base_model", type=str, default="facebook/wav2vec2-large-xlsr-53")
    p.add_argument("--batch_size", type=int, default=2) # Aumentei levemente
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4) # Ajustado para fine-tuning
    p.add_argument("--num_train_epochs", type=int, default=5)
    p.add_argument("--num_workers", type=int, default=2)
    return p.parse_args()

# Global processor para compute_metrics
processor = None

def main():
    global processor
    args = parse_args()
    
    run_dir = Path(args.run_dir).resolve()
    splits_dir = run_dir / "splits"
    
    # Carregar DataFrames
    train_df = pd.read_parquet(splits_dir / "train.parquet")
    valid_df = pd.read_parquet(splits_dir / "valid.parquet")
    # Tokenizer e Feature Extractor
    # Cria vocabulário baseado no treino
    vocab = build_vocab_from_texts(train_df["text_norm"].tolist())
    vocab_dir = run_dir / "vocab"
    vocab_dir.mkdir(parents=True, exist_ok=True)
    
    with open(vocab_dir / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
        
    tokenizer = Wav2Vec2CTCTokenizer(
        str(vocab_dir / "vocab.json"), 
        unk_token="[UNK]", 
        pad_token="[PAD]", 
        word_delimiter_token="|"
    )
    
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        args.base_model, feature_size=1, sampling_rate=TARGET_SR, padding_value=0.0, do_normalize=True, return_attention_mask=True
    )
    
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    
    # Model
    model = Wav2Vec2ForCTC.from_pretrained(
        args.base_model,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        ignore_mismatched_sizes=True
    )
    model.freeze_feature_extractor()

    # Converter pandas para HuggingFace Dataset
    train_ds = Dataset.from_pandas(train_df[["abs_path", "text_norm"]].rename(columns={"abs_path": "audio", "text_norm": "text"}))
    valid_ds = Dataset.from_pandas(valid_df[["abs_path", "text_norm"]].rename(columns={"abs_path": "audio", "text_norm": "text"}))

    #Mudar essa tretamento pos-smokeTest
    print("Processando datasets...")
    train_ds = train_ds.map(prepare_dataset, remove_columns=train_ds.column_names, num_proc=args.num_workers)
    valid_ds = valid_ds.map(prepare_dataset, remove_columns=valid_ds.column_names, num_proc=args.num_workers)

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    training_args = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        group_by_length=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        evaluation_strategy="steps",
        num_train_epochs=args.num_train_epochs,
        fp16=False,
        save_steps=500,
        eval_steps=500,
        logging_steps=100,
        learning_rate=args.lr,
        warmup_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        tokenizer=processor.feature_extractor,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print("Iniciando treino...")
    trainer.train()
    
    final_dir = run_dir / "final_model"
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))
    print(f"Modelo salvo em {final_dir}")

if __name__ == "__main__":
    main()