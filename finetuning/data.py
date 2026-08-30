"""
Carregamento e tokenização do dataset Financial PhraseBank.

Fonte: FinanceMTEB/financial_phrasebank no Hugging Face Hub -- uma versão
em parquet, com split train/test já definido, do dataset original
(Malo et al., 2014, "Good debt or bad debt: Detecting semantic
orientations in economic texts"), subset "sentences_allagree" (todos os
anotadores concordaram na label -- ver DECISOES-FINETUNING.md para porque
este mirror foi escolhido em vez do `takala/financial_phrasebank`
original, que usa um loading script já não suportado pela versão atual
da biblioteca `datasets`).

Labels: 0=negative, 1=neutral, 2=positive.
"""

from __future__ import annotations

from dataclasses import dataclass

from datasets import DatasetDict, load_dataset
from transformers import PreTrainedTokenizerBase

from finetuning.constants import MAX_LENGTH

DATASET_ID = "FinanceMTEB/financial_phrasebank"


@dataclass(frozen=True)
class SentimentExample:
    text: str
    label: int
    label_text: str


def load_raw_dataset() -> DatasetDict:
    """Descarrega (ou usa a cópia em cache local) o Financial PhraseBank."""
    return load_dataset(DATASET_ID)


def tokenize_dataset(
    raw: DatasetDict, tokenizer: PreTrainedTokenizerBase
) -> DatasetDict:
    """Tokeniza os splits train/test para o formato esperado pelo Trainer."""

    def _tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
        )

    tokenized = raw.map(_tokenize, batched=True)
    tokenized = tokenized.remove_columns(["text", "label_text"])
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )
    return tokenized


def load_test_examples() -> list[SentimentExample]:
    """Devolve o split de teste como lista de exemplos simples (para comparação/avaliação)."""
    raw = load_raw_dataset()
    return [
        SentimentExample(
            text=row["text"], label=row["label"], label_text=row["label_text"]
        )
        for row in raw["test"]
    ]


if __name__ == "__main__":
    ds = load_raw_dataset()
    print(ds)
    print("Exemplo train[0]:", ds["train"][0])
    print(
        "Distribuição de labels (train):",
        ds["train"].to_pandas()["label_text"].value_counts().to_dict(),
    )
