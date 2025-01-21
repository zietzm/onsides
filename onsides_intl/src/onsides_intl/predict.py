import logging
from pathlib import Path

import numpy as np
import torch
from pydantic import BaseModel

from onsides_intl.clinicalbert import ClinicalBertClassifier, evaluate
from onsides_intl.stringsearch import IndexedText

logger = logging.getLogger(__name__)


class TextSettings(BaseModel):
    """
    Settings for the text data.

    This is a legacy compatibility thing, and doesn't necessarily need to be
    used if you are just evaluating the model. It is used to check that the
    text data are formatted the same way as the model's training data.
    """

    nwords: int = 125
    refset: int = 14
    section: str = "AR"


class TextPrediction(BaseModel):
    text_id: int
    prediction: float


def predict(
    texts: list[IndexedText],
    network_path: Path,
    weights_path: Path,
    text_settings: TextSettings | None = None,
    batch_size: int | None = None,
    device_id: int | None = None,
) -> list[TextPrediction]:
    """Predict the labels for a list of texts.

    Args:
        texts: formatted context texts (e.g. "headache this drug caused EVENT")
        network_path: path to the model specification
        weights_path: path to the model weights
        text_settings: settings for the text data (number of words, etc.)
        batch_size: batch size for prediction. Defaults to 2x train batch size
        device_id: device ID to use for prediction. "cpu", "cuda:0", etc.

    Returns:
        list[TextPrediction]: prediction values for each text
    """
    if text_settings is None:
        text_settings = TextSettings()

    train_settings = _TrainModelSettings.from_filename(weights_path)
    _validate_settings(train_settings, text_settings)

    if batch_size is None:
        batch_size = train_settings.batch_size * 2

    logger.info(f"Loading model from {network_path}")
    model = ClinicalBertClassifier(network_path)

    logger.info(f"Loading model data from {weights_path}")
    state_dict = torch.load(weights_path, weights_only=True)
    state_dict.pop("bert.embeddings.position_ids")  # Error if included
    model.load_state_dict(state_dict)

    logger.info("Evaluating text with the model...")
    strings = [x.text for x in texts]
    predictions = evaluate(
        model,
        network_path,
        strings,
        max_length=train_settings.max_length,
        batch_size=batch_size,
        device_id=device_id,
    )
    assert len(texts) == len(predictions)
    return [
        TextPrediction(text_id=x.text_id, prediction=y[0])  # The first item ("Pred0")
        for x, y in zip(texts, predictions)
    ]


class _TrainModelSettings(BaseModel):
    prefix: str
    network: str
    refset: int
    refsource: str
    refsection: str
    refnwords: int
    np_random_seed: int
    split_method: str
    epochs: int
    lr: str
    max_length: int
    batch_size: int

    @classmethod
    def from_filename(cls, model_filepath: Path) -> "_TrainModelSettings":
        splits = model_filepath.stem.split("_")
        if len(splits) != 8:
            raise Exception(
                "Model filename not in format expected: {prefix}_{refset}_"
                "{np_random_seed}_{split_method}_{EPOCHS}_{LR}_{max_length}_"
                "{batch_size}.pth"
            )

        prefix = splits[0]
        network = prefix.split("-")[2]
        refset, refsection, refnwords, refsource = splits[1].split("-")

        return cls(
            prefix=prefix,
            network=network,
            refset=int(refset),
            refsource=refsource,
            refsection=refsection,
            refnwords=int(refnwords),
            np_random_seed=int(splits[2]),
            split_method=splits[3],
            epochs=int(splits[4]),
            lr=splits[5],
            max_length=int(splits[6]),
            batch_size=int(splits[7]),
        )

    def __str__(self):
        return (
            "Model\n"
            "-------------------\n"
            f" prefix: {self.prefix}\n"
            f" network: {self.network}\n"
            f" refset: {self.refset}\n"
            f" refsource: {self.refsource}\n"
            f" refsection: {self.refsection}\n"
            f" refnwords: {self.refnwords}\n"
            f" np_random_seed: {self.np_random_seed}\n"
            f" split_method: {self.split_method}\n"
            f" EPOCHS: {self.epochs}\n"
            f" LR: {self.lr}\n"
            f" max_length: {self.max_length}\n"
            f" batch_size: {self.batch_size}\n"
        )


class _ExampleDataSettings(BaseModel):
    filename: str
    refset: int
    nwords: int
    prefix: str
    section: str
    is_split: bool
    split_no: str
    _path: Path

    @classmethod
    def from_filename(cls, examples_path: Path) -> "_ExampleDataSettings":
        splits = examples_path.stem.split("_")
        if len(splits) < 8:
            raise ValueError("Expected 8 splits, got {len(splits)}")

        refset = int(splits[1].removeprefix("method"))
        nwords = int(splits[2].removeprefix("nwords"))
        if "split" in examples_path.name:
            is_split = True
            split_no = "-" + examples_path.name.split("split")[1]
        else:
            is_split = False
            split_no = ""

        return cls(
            filename=examples_path.name,
            refset=refset,
            nwords=nwords,
            prefix=splits[0],
            section=splits[7],
            is_split=is_split,
            split_no=split_no,
            _path=examples_path,
        )

    def __str__(self):
        return (
            "Example Data\n"
            "-------------------\n"
            f" prefix: {self.prefix}\n"
            f" refset: {self.refset}\n"
            f" is_split: {self.is_split}\n"
            f" split_no: {self.split_no}\n"
        )


def _validate_settings(
    model_settings: _TrainModelSettings,
    text_settings: TextSettings | _ExampleDataSettings,
) -> None:
    if text_settings.nwords != model_settings.refnwords:
        raise ValueError(
            f"ERROR: There is an nwords mismatch between the model "
            f"({model_settings.refnwords}) and the example data "
            f"({text_settings.nwords})."
        )
    if text_settings.refset != model_settings.refset:
        raise ValueError(
            "Examples were generated with different reference method "
            f"({text_settings.refset}) than the model was trained with "
            f"({model_settings.refset})"
        )
    if text_settings.section != model_settings.refsection:
        raise ValueError(
            "Examples were generated for a different label section "
            f"({text_settings.section}) than the model was trained with "
            f"({model_settings.refsection})"
        )


def _build_results_path(
    model_settings: _TrainModelSettings,
    example_settings: _ExampleDataSettings,
) -> Path:
    """
    Build the path to the results file for a given model and example data.
    This is the legacy file naming convention.
    """
    filename = (
        f"{model_settings.prefix}-{example_settings.prefix}{example_settings.split_no}_"
        f"ref{model_settings.refset}-{model_settings.refsection}-"
        f"{model_settings.refnwords}-{model_settings.refsource}_"
        f"{model_settings.np_random_seed}_{model_settings.split_method}_"
        f"{model_settings.epochs}_{model_settings.lr}_"
        f"{model_settings.max_length}_{model_settings.batch_size}.csv.gz"
    )
    return example_settings._path.parent / filename
