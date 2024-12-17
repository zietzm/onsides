import argparse
import logging
import os
import pathlib
import sys

import pandas as pd

sys.path.insert(
    0,
    pathlib.Path(__file__)
    .parent.parent.parent.parent.parent.joinpath("src")
    .as_posix(),
)
import predict

logger = logging.getLogger(__name__)


def predict_all(
    data_folder: pathlib.Path,
    model_path: pathlib.Path,
) -> None:
    logger.info("Running the OnSIDES model")
    exact_terms_path = (
        data_folder
        / "sentences-rx_method14_nwords125_clinical_bert_application_set_AR_v0924.csv"
    )
    ar_model = (
        model_path / "bestepoch-bydrug-PMB_14-AR-125-all_222_24_25_2.5e-05_256_32.pth"
    )
    assert ar_model.exists()

    # get absolute path - we want the onsides folder to find and call predict.py
    onsides_intl_dir = pathlib.Path(os.getcwd())
    assert onsides_intl_dir.stem == "onsides_intl"
    onsides_dir = onsides_intl_dir.parent
    script_path = onsides_dir / "src" / "predict.py"
    assert script_path.exists()

    # call the prediction model
    predict.predict(
        model_filepath=ar_model,
        models_path=model_path,
        examples_path=exact_terms_path,
        batch_size=None,
    )
    # command = (
    #     f"python3 {script_path} --model {ar_model.absolute()} "
    #     f"--examples {exact_terms_path.absolute()} "
    #     f"--models_path {model_path.absolute()}"
    # )
    # subprocess.run(shlex.split(command), check=True, cwd=model_path.parent)

    # build files using predicted labels
    result_path = data_folder / (
        "bestepoch-bydrug-PMB-sentences-rx_ref14-AR-125-all_222_24_25_2.5e-"
        "05_256_32.csv.gz"
    )
    assert result_path.exists()

    # right now, we have it set up to simply run through the results and just
    # filter against the threshold used in the original OnSIDES output.
    threshold = 0.4633
    result_df = pd.read_csv(result_path, header=None, names=["Pred0", "Pred1"])
    exact_df = pd.read_csv(exact_terms_path)
    df = pd.concat([exact_df, result_df], axis=1)
    n_before = df.shape[0]
    df = df[df.Pred0 > threshold]
    n_after = df.shape[0]
    logger.info(f"Filtered from {n_before} to {n_after} rows.")
    df.to_csv(data_folder / "ade_text_table_onsides_pred.csv", index=False)


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
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
