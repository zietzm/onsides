import argparse
import logging
import pathlib

import onsides_intl.predict
import pandas as pd

logger = logging.getLogger(__name__)


def predict_all(
    data_folder: pathlib.Path,
    model_path: pathlib.Path,
) -> None:
    logger.info("Loading exact matches...")
    exact_terms_path = (
        data_folder
        / "sentences-rx_method14_nwords125_clinical_bert_application_set_AR_v0924.csv"
    )
    exact_matches_df = pd.read_csv(exact_terms_path)
    strings = [str(x) for x in exact_matches_df["string"].to_list()]

    ar_model = (
        model_path / "bestepoch-bydrug-PMB_14-AR-125-all_222_24_25_2.5e-05_256_32.pth"
    )
    assert ar_model.exists()

    network_path = (
        model_path / "microsoft" / "BiomedNLP-PubMedBERT-base-uncased-abstract/"
    )
    assert network_path.exists()

    text_settings = onsides_intl.predict.TextSettings(
        nwords=125, refset=14, section="AR"
    )
    logger.info("Predicting labels...")
    predictions = onsides_intl.predict.predict(
        texts=strings,
        network_path=network_path,
        weights_path=ar_model,
        text_settings=text_settings,
        batch_size=None,
    )

    logger.info("Saving outputs...")
    # right now, we have it set up to simply run through the results and just
    # filter against the threshold used in the original OnSIDES output.
    threshold = 0.4633
    df = exact_matches_df.assign(Pred0=predictions).loc[
        lambda df: df["Pred0"] > threshold
    ]
    n_before = exact_matches_df.shape[0]
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
