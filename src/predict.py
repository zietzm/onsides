"""
predict.py

Apply a trained model to a set of examples.

@author Nicholas P. Tatonetti, PhD
"""

import argparse
import logging
import os
import pathlib
import sys

import numpy as np
import pandas as pd
import torch
from pydantic import BaseModel

sys.path.append(os.path.abspath("./src"))
import fit_clinicalbert as cb

logger = logging.getLogger(__name__)


def predict(
    model_filepath: pathlib.Path,
    models_path: pathlib.Path,
    examples_path: pathlib.Path,
    batch_size: int | None = None,
) -> None:
    train_settings = TrainModelSettings.from_filename(model_filepath)
    print(train_settings)
    ex_settings = ExampleDataSettings.from_filename(examples_path)

    if ex_settings.nwords != train_settings.refnwords:
        raise ValueError(
            f"ERROR: There is an nwords mismatch between the model "
            f"({train_settings.refnwords}) and the example data "
            f"({ex_settings.nwords})."
        )
    if ex_settings.refset != train_settings.refset:
        raise ValueError(
            "Examples were generated with different reference method "
            f"({ex_settings.refset}) than the model was trained with "
            f"({train_settings.refset})"
        )
    if ex_settings.section != train_settings.refsection:
        raise ValueError(
            "Examples were generated for a different label section "
            f"({ex_settings.section}) than the model was trained with "
            f"({train_settings.refsection})"
        )
    if ex_settings.section != train_settings.refsection:
        logger.warning(
            f"WARNING: The examples section ({ex_settings.section}) does not "
            f"match the reference section ({train_settings.refsection}))"
        )

    is_split = False
    split_no = ""
    if "split" in ex_settings.filename:
        is_split = True
        split_no = "-" + ex_settings.filename.split("split")[1]

    results_filename = (
        f"{train_settings.prefix}-{ex_settings.prefix}{split_no}_"
        f"ref{train_settings.refset}-{train_settings.refsection}-"
        f"{train_settings.refnwords}-{train_settings.refsource}_"
        f"{train_settings.np_random_seed}_{train_settings.split_method}_"
        f"{train_settings.epochs}_{train_settings.lr}_"
        f"{train_settings.max_length}_{train_settings.batch_size}.csv.gz"
    )
    results_path = examples_path.parent / results_filename
    if results_path.exists():
        raise FileExistsError(
            f"Results file already exists: {results_path}"
            "To re-generate the results, delete the file and try again."
        )

    print("Examples")
    print("-------------------")
    print(f" prefix: {ex_settings.prefix}")
    print(f" refset: {ex_settings.refset}")
    print(f" is_split: {is_split}")
    print(f" split_no: {split_no}")
    print(f" results path: {results_path}\n")

    if batch_size is not None:
        logger.info(f"Overriding batch_size to user defined: {batch_size}")
    else:
        # default is to use 2X the training batch size
        batch_size = train_settings.batch_size * 2
        logger.info(
            "Setting prediction batch_size to 2X the training batch_size: "
            f"{batch_size}"
        )

    if train_settings.network.startswith("CB"):
        network_path = models_path / "Bio_ClinicalBERT"
    elif train_settings.network.startswith("PMB"):
        network_path = (
            models_path / "microsoft" / "BiomedNLP-PubMedBERT-base-uncased-abstract"
        )
    else:
        raise ValueError(f"ERROR: Unknown network: {train_settings.network}")

    # initialize Dataset.tokenizer
    cb.Dataset.set_tokenizer(network_path)
    model = cb.ClinicalBertClassifier(network_path)
    logger.info(f"Loading model from {model_filepath.name}")
    state_dict = torch.load(model_filepath, weights_only=True)
    state_dict.pop("bert.embeddings.position_ids")  # Error if included
    model.load_state_dict(state_dict)

    examples_df = pd.read_csv(examples_path)

    logger.info("Evaluating example data...")
    outputs = cb.evaluate(
        model, examples_df, train_settings.max_length, batch_size, examples_only=True
    )
    npoutputs = [x.cpu().detach().numpy() for x in outputs]
    predictions = np.vstack(npoutputs)
    np.savetxt(results_path, predictions, delimiter=",")


class TrainModelSettings(BaseModel):
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
    def from_filename(cls, model_filepath: pathlib.Path) -> "TrainModelSettings":
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


class ExampleDataSettings(BaseModel):
    filename: str
    refset: int
    nwords: int
    prefix: str
    section: str

    @classmethod
    def from_filename(cls, examples_path: pathlib.Path) -> "ExampleDataSettings":
        splits = examples_path.stem.split("_")
        if len(splits) < 8:
            raise ValueError("Expected 8 splits, got {len(splits)}")

        refset = int(splits[1].removeprefix("method"))
        nwords = int(splits[2].removeprefix("nwords"))
        return cls(
            filename=examples_path.name,
            refset=refset,
            nwords=nwords,
            prefix=splits[0],
            section=splits[7],
        )


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", help="path to the model (pth) file", type=pathlib.Path, required=True
    )
    parser.add_argument(
        "--models_path",
        type=pathlib.Path,
        required=True,
        help="path to the model to use",
    )
    parser.add_argument(
        "--examples",
        help="path to the file that contains the examples to make predict for",
        type=pathlib.Path,
        required=True,
    )
    parser.add_argument(
        "--batch-size", default=None, type=int, help="override the default batch size"
    )
    args = parser.parse_args()
    predict(
        model_filepath=args.model,
        models_path=args.models_path,
        examples_path=args.examples,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
