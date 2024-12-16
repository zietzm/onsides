import argparse
import logging
import os
import pathlib

import pandas as pd

logger = logging.getLogger(__name__)


def predict_all(
    data_folder: pathlib.Path,
    model_path: pathlib.Path,
) -> None:
    logger.info("Running the OnSIDES model")

    exact_terms_path = data_folder / "bert_input.csv"
    ar_model = (
        model_path
        / data_folder
        / "bestepoch-bydrug-PMB_14-AR-125-all_222_24_25_2.5e-05_256_32.pth"
    )

    # get absolute path - we want the onsides folder to find and call the predict.py script
    onsides_intl_dir = os.path.abspath(os.getcwd())
    onsides_dir = onsides_intl_dir.replace("onsides_intl/", "")

    # call the prediction model
    os.system(
        f"python3 {onsides_dir}src/predict.py --model {ar_model} --examples {exact_terms_path}"
    )

    # build files using predicted labels
    # TODO : customize the create_onsides_datafiles script for the EU data
    results = (
        data_folder
        / "bestepoch-bydrug-PMB-sentences-rx_ref14-AR-125-all_222_24_25_2.5e-05_256_32.csv.gz"
    )

    # right now, we have it set up to simply run through the results and just filter against the threshold used in the original OnSIDES output.
    res = results
    ex = exact_terms_path
    threshold = 0.4633
    res = pd.read_csv(res, header=None, names=["Pred0", "Pred1"])
    ex = pd.read_csv(ex)
    df = pd.concat([ex, res], axis=1)
    print(df.shape[0])
    df = df[df.Pred0 > threshold]
    print(df.shape[0])
    df.to_csv(data_folder / "data/ade_text_table_onsides_pred.csv", index=False)


def main():
    parser = argparse.ArgumentParser(
        description="let the code know where the data is held"
    )
    parser.add_argument(
        "--data_folder",
        type=pathlib.Path,
        required=True,
        help="Path to the data folder.",
    )
    parser.add_argument(
        "--model_path",
        type=pathlib.Path,
        required=True,
        help="Path to the where the model is housed.",
    )
    args = parser.parse_args()
    data_folder = args.data_folder
    model_path = args.model_path
    predict_all(data_folder, model_path)


if __name__ == "__main__":
    main()
