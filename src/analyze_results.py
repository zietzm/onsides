import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.abspath("./src"))
import fit_clinicalbert as cb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the model to construct predictions from.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        default=True,
        help="Skip generating predictions for the training data (which can take a long time)",
    )
    parser.add_argument(
        "--network",
        help="path to pretained network, default is 'models/Bio_ClinicalBERT'",
        type=str,
        default="models/Bio_ClinicalBERT/",
    )
    parser.add_argument("--base-dir", type=str, default=".")

    args = parser.parse_args()

    model_filepath = args.model
    model_file = os.path.split(model_filepath)[-1]

    print(f"Loading model from {model_file}")
    # fnnoext = os.path.split(model_file)[-1].split('.')[0]
    fnnoext = os.path.splitext(os.path.split(model_file)[-1])[0]

    if len(fnnoext.split("_")) not in (6, 8):
        raise Exception(
            f"Model filename not in format expected: "
            f"{prefix}_{refset}_{np_random_seed}_{split_method}_{EPOCHS}_{LR}.pth"
            f" or {prefix}_{refset}_{np_random_seed}_{split_method}_{EPOCHS}_"
            f"{LR}_{MAX_LENGTH}_{BATCH_SIZE}.pth"
        )

    refset, refsection, refnwords, refsource = fnnoext.split("_")[1].split("-")
    np_random_seed = int(fnnoext.split("_")[2])
    split_method = fnnoext.split("_")[3]
    EPOCHS = int(fnnoext.split("_")[4])
    LR = fnnoext.split("_")[5]

    if len(fnnoext.split("_")) == 8:
        max_length = int(fnnoext.split("_")[6])
        batch_size = int(fnnoext.split("_")[7])
    else:
        max_length = 128
        batch_size = 128

    prefix = fnnoext.split("_")[0]

    _, network_path, _ = cb.parse_network_argument(args.network)

    print(f" prefix: {prefix}")
    print(f" refset: {refset}")
    print(f" np_random_seed: {np_random_seed}")
    print(f" split_method: {split_method}")
    print(f" EPOCHS: {EPOCHS}")
    print(f" LR: {LR}")
    print(f" max_length: {max_length}")
    print(f" batch_size: {batch_size}")
    print(f" skip_train?: {args.skip_train}")

    # load pre-trained network
    cb.Dataset.set_tokenizer(network_path)
    model = cb.ClinicalBertClassifier(network_path)
    state = torch.load(model_filepath)
    state.pop("bert.embeddings.position_ids")
    model.load_state_dict(state)

    # loading and re-splitting the data
    datapath = f"./data/refs/ref{refset}_nwords{refnwords}_clinical_bert_reference_set_{refsection}.txt"
    if not os.path.exists(datapath):
        raise Exception(f"ERROR: No reference set file found at {datapath}")

    df = cb.load_reference_data(datapath, refsource)

    df_train, df_val, df_test = cb.split_train_val_test(
        df, np_random_seed, split_method
    )

    file_parameters = f"{refset}-{refsection}-{refnwords}-{refsource}_{np_random_seed}_{split_method}_{EPOCHS}_{LR}_{max_length}_{batch_size}"

    test_filename = f"{args.base_dir}/results/{prefix}-test_{file_parameters}.csv"

    print(f"Evaluating testing data, will save to: {test_filename}")
    print(f"\t df_test.shape = {df_test.shape}")
    outputs = cb.evaluate(model, df_test, max_length, batch_size)
    npoutputs = [x.cpu().detach().numpy() for x in outputs]
    predictions = np.vstack(npoutputs)

    np.savetxt(test_filename, predictions, delimiter=",")

    valid_filename = f"{args.base_dir}/results/{prefix}-valid_{file_parameters}.csv"

    print(f"Evaluating validation data, will save to: {valid_filename}")
    print(f"\t df_val.shape = {df_val.shape}")
    outputs = cb.evaluate(model, df_val, max_length, batch_size)
    npoutputs = [x.cpu().detach().numpy() for x in outputs]
    predictions = np.vstack(npoutputs)

    np.savetxt(valid_filename, predictions, delimiter=",")
    assert df_val.shape[0] == predictions.shape[0], (
        "Validation predictions not the expected shape"
    )
    valid_df_filename = (
        f"{args.base_dir}/results/{prefix}-valid_{file_parameters}-full.csv"
    )
    df_val.to_csv(valid_df_filename, index=False)

    if not args.skip_train:
        train_filename = f"{args.base_dir}/results/{prefix}-train_{file_parameters}.csv"

        print(f"Evaluating training data, will save to: {train_filename}")
        print(f"\t df_train.shape = {df_train.shape}")
        outputs = cb.evaluate(model, df_train, max_length, batch_size)
        npoutputs = [x.cpu().detach().numpy() for x in outputs]
        predictions = np.vstack(npoutputs)

        np.savetxt(train_filename, predictions, delimiter=",")
