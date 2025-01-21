"""
fit_clinicalbert.py

Use clinical bert to classify terms as events or not_events.

@author Nicholas Tatonetti, Tatonetti Lab (inspired by https://towardsdatascience.com/text-classification-with-bert-in-pytorch-887965e5820f)
"""

import argparse
import csv
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

labels = {"not_event": 0, "is_event": 1}
pretrained_state_ids = {
    "bestepoch-bydrug-CB_0-AR-125-all_222_24_25_1e-06_256_32.pth": "0",
    "bestepoch-bydrug-CB_0-BW-125-all_222_24_25_1e-06_256_32.pth": "1",
    "bestepoch-bydrug-CB_0-ALL-125-all_222_24_25_1e-06_256_32.pth": "2",
    "bestepoch-bydrug-PMB_14-AR-125-all_222_24_25_1e-06_256_32.pth": "0",
    "bestepoch-bydrug-PMB_14-BW-125-all_222_24_25_1e-06_256_32.pth": "1",
    "bestepoch-bydrug-PMB_14-ALL-125-all_222_24_25_1e-06_256_32.pth": "2",
}


class Dataset(torch.utils.data.Dataset):
    tokenizer = None

    @staticmethod
    def set_tokenizer(pretrained_model_path):
        print(f"Loading tokenizer from {pretrained_model_path}...")
        Dataset.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)

    def __init__(self, df, examples_only=False, _max_length=128):
        if Dataset.tokenizer is None:
            raise Exception(
                "ERROR: The tokenizer has not yet been initailized. Initialize with Dataset.set_tokenizer(...) before instantiating a class."
            )

        if not examples_only:
            self.labels = [labels[label] for label in df["class"]]
        else:
            self.labels = [0 for _ in range(len(df))]

        self.texts = [
            Dataset.tokenizer(
                text,
                padding="max_length",
                max_length=_max_length,
                truncation=True,
                return_tensors="pt",
            )
            for text in df["string"]
        ]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y


class ClinicalBertClassifier(nn.Module):
    def __init__(self, pretrained_model_path, dropout=0.5):
        super(ClinicalBertClassifier, self).__init__()

        self.bert = AutoModel.from_pretrained(pretrained_model_path)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 2)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(
            input_ids=input_id, attention_mask=mask, return_dict=False
        )
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer


def train(
    model,
    train_data,
    val_data,
    learning_rate,
    epochs,
    max_length,
    batch_size,
    model_filename,
):
    skip_training = False
    if epochs == 0:
        skip_training = True
        epochs = 1

    model.train()

    train, val = (
        Dataset(train_data, _max_length=max_length),
        Dataset(val_data, _max_length=max_length),
    )

    train_dataloader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=True
    )
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=batch_size)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    print(f"Using device: {device}")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    best_val_loss = None
    train_accuracies = list()
    train_losses = list()
    valid_accuracies = list()
    valid_losses = list()
    epoch_times = list()
    epoch_saved = list()
    epochs_since_best = 0

    for epoch_num in range(epochs):
        total_acc_train = 0
        total_loss_train = 0
        saved_model = False
        epoch_start_time = time.time()
        epochs_since_best += 1

        for train_input, train_label in tqdm(train_dataloader):
            if skip_training:
                break

            train_label = train_label.to(device)
            mask = train_input["attention_mask"].to(device)
            input_id = train_input["input_ids"].squeeze(1).to(device)

            output = model(input_id, mask)

            batch_loss = criterion(output, train_label)
            total_loss_train += batch_loss.item()

            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():
            for val_input, val_label in val_dataloader:
                val_label = val_label.to(device)
                mask = val_input["attention_mask"].to(device)
                input_id = val_input["input_ids"].squeeze(1).to(device)

                output = model(input_id, mask)

                batch_loss = criterion(output, val_label)
                total_loss_val += batch_loss.item()

                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc

            if (
                best_val_loss is None
                or (total_loss_val / len(val_data)) < best_val_loss
            ):
                # best epoch so far, we save it to file
                best_val_loss = total_loss_val / len(val_data)
                torch.save(model.state_dict(), model_filename)
                saved_model = True
                epochs_since_best = 0

        train_losses.append(total_loss_train / len(train_data))
        train_accuracies.append(total_acc_train / len(train_data))
        valid_losses.append(total_loss_val / len(val_data))
        valid_accuracies.append(total_acc_val / len(val_data))
        epoch_times.append(time.time() - epoch_start_time)
        epoch_saved.append(saved_model)

        print(f"Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .4f} \
                | Train Accuracy: {total_acc_train / len(train_data): .4f} \
                | Val Loss: {total_loss_val / len(val_data): .4f} \
                | Val Accuracy: {total_acc_val / len(val_data): .4f}")

        print(
            f"It's been {epochs_since_best} since the best performing epoch. Will break if this hits 4."
        )
        if epochs_since_best >= 4:
            print(f"  Stopping here.")
            return (
                train_losses,
                train_accuracies,
                valid_losses,
                valid_accuracies,
                epoch_times,
                epoch_saved,
            )

    return (
        train_losses,
        train_accuracies,
        valid_losses,
        valid_accuracies,
        epoch_times,
        epoch_saved,
    )


def evaluate(model, test_data, max_length, batch_size, examples_only=False):
    model.eval()
    test = Dataset(test_data, examples_only, _max_length=max_length)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=batch_size)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        model = model.cuda()

    total_acc_test = 0
    outputs = list()

    with torch.no_grad():
        for test_input, test_label in tqdm(test_dataloader):
            test_label = test_label.to(device)
            mask = test_input["attention_mask"].to(device)
            input_id = test_input["input_ids"].squeeze(1).to(device)

            output = model(input_id, mask)
            outputs.append(output)

            acc = (output.argmax(dim=1) == test_label).sum().item()
            total_acc_test += acc

    if not examples_only:
        print(f"Test Accuracy: {total_acc_test / len(test_data): .4f}")

    return outputs


def batch_size_estimate(max_length):
    log_bs = -1.2209302325581395 * np.log(max_length) + 10.437506963082898
    bs = np.exp(log_bs)
    power = np.log2(bs)
    return 2 ** round(power)


def split_train_val_test(df, np_random_seed, split_method):
    # randomly select by drug/label
    druglist = sorted(set(df["drug"]))
    # print(druglist)
    random.seed(np_random_seed)
    random.shuffle(druglist)
    # print(druglist)

    np.random.seed(np_random_seed)
    if split_method == "TAC":
        # split accoding to TAC training/testing splits
        if not "tac" in df:
            raise Exception(
                "ERROR: The 'tac' column was not found in the reference data. This is likely due to the ref file being generated using an old version of create_training_data.py. Regenerating using construct_training_data.py should fix this issue."
            )

        train_drugs = sorted(set(df[df["tac"] == "train"]["drug"]))
        drugs_train, drugs_val = np.split(train_drugs, [int(0.9 * len(train_drugs))])
        drugs_test = sorted(set(df[df["tac"] == "test"]["drug"]))

    elif int(split_method) == 24:
        # default approach, use all drugs split 80/10/10
        drugs_train, drugs_val, drugs_test = np.split(
            druglist, [int(0.8 * len(druglist)), int(0.9 * len(druglist))]
        )

    print(f"Split labels in train, val, test by drug:")
    print(len(drugs_train), len(drugs_val), len(drugs_test))

    df_train = df[df["drug"].isin(drugs_train)]
    df_val = df[df["drug"].isin(drugs_val)]
    df_test = df[df["drug"].isin(drugs_test)]

    print(f"Starting examples has shape: {df.shape}")
    print(f"Resulting examples have shapes: ")
    print(f"\t df_train.shape = {df_train.shape}")
    print(f"\t df_val.shape = {df_val.shape}")
    print(f"\t df_test.shape = {df_test.shape}")

    return df_train, df_val, df_test


def parse_network_argument(args_network):
    network_path = None
    pretrained_state = None
    if args_network.endswith(".pth"):
        # load a previoulsy saved state
        pretrained_state = args_network
        network_code = os.path.split(args_network)[-1].split("_")[0].split("-")[-1]
        if network_code == "CB":
            network_path = "./models/Bio_ClinicalBERT/"
        elif network_code == "PMB":
            network_path = (
                "./models/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract/"
            )
        else:
            raise Excepction(
                f"ERROR: Pretrained state has unexpected network code: {network_code}"
            )

        if not os.path.split(pretrained_state)[-1] in pretrained_state_ids:
            raise Exception(
                f"ERROR: The pretrained state you are using ({pretrained_state}) does not have an identifer. Add it to the dictionary in fit_clinicalbert.py."
            )
        network_code += pretrained_state_ids[os.path.split(pretrained_state)[-1]]

    elif args_network.find("Bio_ClinicalBERT") != -1:
        # ClinicalBert
        network_code = "CB"
        network_path = "./models/Bio_ClinicalBERT/"
    elif args_network.find("BiomedNLP-PubMedBERT") != -1:
        # PubMedBert
        network_code = "PMB"
        network_path = "./models/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract/"
    else:
        raise Exception(f"ERROR: Unexpected pretrained model: {args_network}")

    return network_code, network_path, pretrained_state


def load_reference_data(datapath, source):
    df = pd.read_csv(datapath)

    if not "source_method" in df:
        if source != "exact":
            raise Exception(
                f"ERROR: Invalid source ({source}) for old reference file format. Must be 'exact'."
            )
        return df

    if source == "all":
        # nothing to do here
        pass
    elif source in ("exact", "deepcadrme"):
        df = df[df["source_method"] == source]
    else:
        raise Exception(
            f"ERROR: Unexpected value passed in for --refsource argument: {source}"
        )

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ref",
        help="relative or full path to the reference set",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--max-length",
        help="maximum number of tokens to use as input for the BERT model, default is smallest power of two that is greater than 2*nwords used in the construction of the reference set",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--batch-size",
        help="batch size to feed into the model each epoch, will need to balance with max_length to avoid memory errors, default is estimated from a set of runs we've previously run and should work alright",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--epochs",
        help="number of epochs to train, default is 25",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--learning-rate",
        help="the learning rate to use, default is 1e-6",
        type=float,
        default=1e-6,
    )
    parser.add_argument(
        "--ifexists",
        help="what to do if model already exists with same parameters, options are 'replicate', 'overwrite', 'quit' - default is 'quit'",
        type=str,
        default="quit",
    )
    parser.add_argument(
        "--network",
        help="path to pretained network, default is 'models/Bio_ClinicalBERT', but you can use other pretrained models or you can use previously saved states.",
        type=str,
        default="models/Bio_ClinicalBERT/",
    )
    parser.add_argument("--base-dir", type=str, default=".")
    parser.add_argument(
        "--refsource",
        help="restrict reference data to only one source type, values may be 'all', 'exact', or 'deepcadrme'.",
        type=str,
        default="all",
    )
    parser.add_argument(
        "--split-method",
        help="which method to use to split the data into train/valid/test, default is 90/10/10 and possible values are 24 or TAC",
        type=str,
        default="24",
    )
    args = parser.parse_args()

    print(f"Loading reference data...")

    # datapath = './data/clinical_bert_reference_set.txt'
    df = load_reference_data(args.ref, args.refsource)

    print(df.head())
    print(len(df))

    print("Splitting data into training, validation, and testing...")
    refset = int(os.path.basename(args.ref).split("ref")[1].split("_")[0])
    refsection = os.path.basename(args.ref).split("_")[-1].split(".")[0]
    refnwords = int(os.path.basename(args.ref).split("nwords")[1].split("_")[0])
    refsource = args.refsource

    print(
        f"Reference set loaded from {args.ref}\n\tmethod: {refset}\n\tsection: {refsection}\n\tnwords: {refnwords}\n\trefsource: {refsource}"
    )

    np_random_seed = 222

    # training/validation/testing split method
    # 24 => Default, randomly split by drugs 80/10/10
    # TAC => training/validation split 90/10 from tac training data (101 labels)
    #        testing is 100 of tac testing data (99 labels)
    split_method = args.split_method

    max_length = None
    batch_size = args.batch_size
    EPOCHS = args.epochs
    LR = args.learning_rate

    network_code, network_path, pretrained_state = parse_network_argument(args.network)

    if args.max_length == -1:
        # Default option, we set it to smallest power of two greater than
        # 2*nwords in the reference set. In our analysis we found that
        # the number of tokens per example was on average 1.7-1.8X the number
        # of words. And that 2X would capture the entire example string for
        # about 75% of examples.
        max_length = 2 ** int(np.ceil(np.log2(2 * refnwords)))
        print(
            f" Reference set used {refnwords} nwords per example, max_length is set to {max_length}"
        )
    else:
        max_length = args.max_length
        if max_length < 2 * refnwords:
            print(
                f" WARNING: max_length is set to less than 2*nrefwords, there will be truncatation for more than 25% of example strings."
            )

    if args.batch_size == -1:
        # Default option, we set it using a function we fit on the previous
        # runs that haven't run into any memory errors. We found a line
        # that fits in log-log space with an r^2 = 0.986
        # NOTE: This is machine dependent! We are using P100s with 16GB of memory
        batch_size = batch_size_estimate(max_length)
        print(
            f" Based on the max_length, we are estimating that a batch_size of {batch_size} is the largest that will not run into memory issues."
        )
    else:
        batch_size = args.batch_size
        est_batch_size = batch_size_estimate(max_length)
        if batch_size > est_batch_size:
            print(
                f" WARNING: the provided batch size ({batch_size}) is greater than what we would estimate ({est_batch_size}) will work. You may run into memory issues. If so, reduce the batch size or use the default option value."
            )

    # check for existing model file
    filename_params = f"{refset}-{refsection}-{refnwords}-{refsource}_{np_random_seed}_{split_method}_{EPOCHS}_{LR}_{max_length}_{batch_size}"
    final_model_filename = (
        f"{args.base_dir}/models/final-bydrug-{network_code}_{filename_params}.pth"
    )
    if os.path.exists(final_model_filename):
        print(f"Found final model already saved at path: {final_model_filename}")
        if args.ifexists == "quit":
            print("  Quitting. To run a replicate, use: --ifexists replicate")
            sys.exit(1)
        elif args.ifexists == "replicate":
            print("  Will run a replicate, checking for any existing replicates...")
            reps = [
                f
                for f in os.listdir(f"{args.base_dir}/models/")
                if f.find(filename_params) != -1 and f.lower().find("bestepoch") == -1
            ]
            filename_params = f"{filename_params}_rep{len(reps)}"
            final_model_filename = f"{args.base_dir}/models/final-bydrug-{network_code}_{filename_params}.pth"
            print(
                f"    Found {len(reps)} existing models. Filename for this replicate will be: {final_model_filename}"
            )
        elif args.ifexists == "overwrite":
            print("  Option is to overwrite the exising model file.")
            confirm = input(
                "!!! Please confirm that you would really like to overwrite the existing file? [y/N]"
            )
            if confirm != "y":
                print("  Okay, will not overwrite the file. Quitting instead.")
                sys.exit(1)
        else:
            raise Exception(
                f"ERROR: Unexpected option set for --ifexists argument: {args.ifexists}"
            )

    df_train, df_val, df_test = split_train_val_test(df, np_random_seed, split_method)

    print(f"Resulting dataframes have sizes:")
    print(len(df_train), len(df_val), len(df_test))

    # initailize Dataset.tokenizer
    Dataset.set_tokenizer(network_path)

    # now we can initailize a model
    model = ClinicalBertClassifier(network_path)

    if not pretrained_state is None:
        print(f"Loading pretrained state from model at: {pretrained_state}")
        model.load_state_dict(torch.load(pretrained_state))

    print("Fitting the model...")
    best_epoch_model_filename = (
        f"{args.base_dir}/models/bestepoch-bydrug-{network_code}_{filename_params}.pth"
    )

    # Set the random seed so the results are exactly reproducible
    # Random seeds for the system and numpy are set when before they are used
    torch.manual_seed(np_random_seed)

    training_results = train(
        model,
        df_train,
        df_val,
        LR,
        EPOCHS,
        max_length,
        batch_size,
        best_epoch_model_filename,
    )

    print("Saving the model to file...")

    torch.save(model.state_dict(), final_model_filename)

    print("Saving loss and accuracies for each epoch to file...")
    lafh = open(
        f"{args.base_dir}/results/epoch-results-{network_code}_{filename_params}.csv",
        "w",
    )
    writer = csv.writer(lafh)
    writer.writerow(
        [
            "epoch",
            "train_loss",
            "train_accuracy",
            "valid_loss",
            "valid_accuracy",
            "epoch_time",
            "epoch_saved",
        ]
    )
    for epoch in range(len(training_results[0])):
        writer.writerow(
            [epoch + 1]
            + [training_results[i][epoch] for i in range(len(training_results))]
        )
    lafh.close()

    print("Loading the model from file...")

    loaded_model = ClinicalBertClassifier(network_path)
    loaded_model.load_state_dict(torch.load(final_model_filename))

    print("Evaluating the model on the held out test set...")
    evaluate(loaded_model, df_test, max_length, batch_size)
