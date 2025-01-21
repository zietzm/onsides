import argparse
import logging
import pathlib

import onsides_intl.predict
import pandas as pd

logger = logging.getLogger(__name__)


def predict_all(
    data_folder: pathlib.Path,
    model_path: pathlib.Path,
    batch_size: int | None = None,
    device_id: int | None = None,
) -> None:
    logger.info("Loading exact matches...")
    exact_terms_path = (
        data_folder
        / "sentences-rx_method14_nwords125_clinical_bert_application_set_AR_v0924.csv"
    )
    exact_matches_df = pd.read_csv(exact_terms_path)
    strings = [str(x) for x in exact_matches_df["string"].to_list()]
    logger.info(f"Loaded {len(strings)} strings.")

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
        batch_size=batch_size,
        device_id=device_id,
    )
    predictions_df = pd.DataFrame(predictions, columns=pd.Index(["Pred0", "Pred1"]))

    logger.info("Saving outputs...")
    # right now, we have it set up to simply run through the results and just
    # filter against the threshold used in the original OnSIDES output.
    threshold = 0.4633
    df = pd.concat([exact_matches_df, predictions_df], axis=1).loc[
        lambda df: df["Pred0"] > threshold
    ]
    n_before = exact_matches_df.shape[0]
    n_after = df.shape[0]
    logger.info(f"Filtered from {n_before} to {n_after} rows.")
    df.to_csv(data_folder / "ade_text_table_onsides_pred.csv", index=False)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s.%(msecs)03d %(levelname)s %(module)s -"
            " %(funcName)s: %(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )
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
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="The batch size to use. If None, will use the default.",
    )
    parser.add_argument(
        "--device_id",
        type=int,
        default=None,
        help="The device ID to use. If None, will use the first available.",
    )
    args = parser.parse_args()
    data_folder = args.data_folder
    model_path = args.model_path
    predict_all(data_folder, model_path, args.batch_size, args.device_id)


if __name__ == "__main__":
    main()
