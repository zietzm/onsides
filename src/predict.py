"""
predict.py

Apply a trained model to a set of examples.

@author Nicholas P. Tatonetti, PhD

"""

import argparse
import os
import pathlib
import sys

import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.abspath("./src"))
import fit_clinicalbert as cb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", help="path to the model (pth) file", type=str, required=True
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
        type=str,
        required=True,
    )
    parser.add_argument(
        "--batch-size", default=None, type=int, help="override the default batch size"
    )
    args = parser.parse_args()

    model_filepath = args.model
    model_file = os.path.split(model_filepath)[-1]

    print(f"Loading model from {model_file}")
    model_file_noext = os.path.splitext(model_file)[0]

    if len(model_file_noext.split("_")) != 8:
        raise Exception(
            "Model filename not in format expected: {prefix}_{refset}_"
            "{np_random_seed}_{split_method}_{EPOCHS}_{LR}_{max_length}_"
            "{batch_size}.pth"
        )

    refset, refsection, refnwords, refsource = model_file_noext.split("_")[1].split("-")
    np_random_seed = int(model_file_noext.split("_")[2])
    split_method = model_file_noext.split("_")[3]
    EPOCHS = int(model_file_noext.split("_")[4])
    LR = model_file_noext.split("_")[5]
    max_length = int(model_file_noext.split("_")[6])
    batch_size = int(model_file_noext.split("_")[7])
    prefix = model_file_noext.split("_")[0]
    network = prefix.split("-")[2]

    print(f"Model")
    print(f"-------------------")
    print(f" prefix: {prefix}")
    print(f" network: {network}")
    print(f" refset: {refset}")
    print(f" refsource: {refsource}")
    print(f" refsection: {refsection}")
    print(f" refnwords: {refnwords}")
    print(f" np_random_seed: {np_random_seed}")
    print(f" split_method: {split_method}")
    print(f" EPOCHS: {EPOCHS}")
    print(f" LR: {LR}")
    print(f" max_length: {max_length}")
    print(f" batch_size: {batch_size}\n")

    ex_filename = os.path.split(args.examples)[-1].split(".")[0]
    ex_refset = int(ex_filename.split("_")[1].strip("method"))
    ex_nwords = ex_filename.split("_")[2].strip("nwords")
    ex_prefix = ex_filename.split("_")[0]
    ex_section = ex_filename.split("_")[7]

    if ex_nwords != refnwords:
        raise Exception(
            f"ERROR: There is an nwords mismatch between the model ({refnwords}) and the example data ({ex_nwords})."
        )

    if ex_section != refsection:
        print(
            f"WARNING: The examples section ({ex_section}) does not match the reference section ({refsection}))"
        )

    is_split = False
    split_no = ""
    if ex_filename.find("split") != -1:
        is_split = True
        split_no = "-" + ex_filename.split("split")[1]

    if int(ex_refset) != int(refset):
        raise Exception(
            f"Examples were generated with different reference method ({ex_refset}) than the model was trained with ({refset})"
        )

    if str(ex_section) != str(refsection):
        raise Exception(
            f"Examples were generated for a different label section ({ex_section}) than the model was trained with ({refsection})"
        )

    results_fn = f"{prefix}-{ex_prefix}{split_no}_ref{refset}-{refsection}-{refnwords}-{refsource}_{np_random_seed}_{split_method}_{EPOCHS}_{LR}_{max_length}_{batch_size}.csv.gz"
    examples_dir = os.path.dirname(args.examples)

    results_path = os.path.join(examples_dir, results_fn)

    print(f"Examples")
    print(f"-------------------")
    print(f" prefix: {ex_prefix}")
    print(f" refset: {ex_refset}")
    print(f" is_split: {is_split}")
    print(f" split_no: {split_no}")
    print(f" results path: {results_path}\n")

    if args.batch_size is None:
        # default is to use 2X the training batch size
        batch_size *= 2
        print(
            f"Setting prediction batch_size to 2X the training batch_size: {batch_size}"
        )
    else:
        batch_size = args.batch_size
        print(f"Overriding batch_size to user defined: {batch_size}")

    if os.path.exists(results_path):
        print(
            "  > Results file already exists, will not repeat evaluation. "
            "If you want to re-generate the results, delete the file and try again."
        )
        sys.exit(1)

    if network.startswith("CB"):
        network_path = args.models_path / "Bio_ClinicalBERT/"
    elif network.startswith("PMB"):
        network_path = (
            args.models_path / "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract/"
        )
    else:
        raise Exception(f"ERROR: Unknown network: {network}")

    # initailize Dataset.tokenizer
    cb.Dataset.set_tokenizer(network_path)
    model = cb.ClinicalBertClassifier(network_path)
    state_dict = torch.load(model_filepath)
    state_dict.pop("bert.embeddings.position_ids")
    print(state_dict.keys())
    model.load_state_dict(state_dict)

    # loading the example data
    df = pd.read_csv(args.examples)

    print("Evaluating example data...")
    outputs = cb.evaluate(model, df, max_length, batch_size, examples_only=True)
    npoutputs = [x.cpu().detach().numpy() for x in outputs]
    predictions = np.vstack(npoutputs)

    np.savetxt(results_path, predictions, delimiter=",")
