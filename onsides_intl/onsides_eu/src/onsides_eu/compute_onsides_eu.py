import argparse
import logging
import pathlib
import sys

from onsides_eu import (
    map_drugs_to_rxnorm,
    parse_files,
    tabular_data_mapping,
    text_data_format,
    text_data_parse,
    text_data_predict,
)

logger = logging.getLogger(__name__)


def compute_onsides(
    data_folder: pathlib.Path, external_data_folder: pathlib.Path, model: pathlib.Path
) -> None:
    # Step 0: check if the drug data has been downloaded. If not, return an error
    drug_data_table = data_folder.joinpath("medicines-output-medicines-report_en.xlsx")
    if not drug_data_table.exists():
        logger.error("drug data not found. Please run download_files.py first.")
        sys.exit(1)

    # Step 1: parse the drug data
    logger.info("Parsing text and tables from drug label PDFs")
    parse_files.parse_all_files(data_folder)
    logger.info("Finished parsing text and tables from PDFs")

    # Step 2: extract ades from tabular drug data
    logger.info("Extracting ADEs from tabular drug data")
    tabular_data_mapping.map_tabular(data_folder, external_data_folder)
    logger.info("Finished extracting ADEs from tabular drug data")

    # Step 3: parse pdf files
    logger.info("Parsing text extracted from PDF files")
    text_data_parse.parse_all_text(data_folder)
    logger.info("Finished parsing text")

    # Step 4: extract ades from free text data
    logger.info("Extracting ADEs from free text data")
    text_data_format.format_text(data_folder, external_data_folder)
    logger.info("Finished extracting ADEs from free text data")

    # Step 5: predict the ades from the free text data using the OnSIDES model
    logger.info("Predicting ADEs from free text data")
    # TODO: THIS IS NOT DONE
    text_data_predict.predict_all(data_folder, model)
    logger.info("Finished predicting ADEs from free text data")

    # Step 6: map all of the drugs to standard rxnorm
    # TODO: THIS IS NOT DONE
    logger.info("Mapping drugs to RxNorm")
    map_drugs_to_rxnorm.map_all(data_folder, external_data_folder)
    logger.info("Finished mapping drugs to RxNorm")
    logger.info("Finished all steps!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_folder",
        type=pathlib.Path,
        required=True,
        help="Path to the data folder.",
    )
    parser.add_argument(
        "--external_data",
        type=pathlib.Path,
        required=True,
        help="Path to the where the external data is housed.",
    )
    parser.add_argument(
        "--model",
        type=pathlib.Path,
        required=True,
        help="Path to the where the model is housed.",
    )
    args = parser.parse_args()
    compute_onsides(args.data_folder, args.external_data, args.model)


if __name__ == "__main__":
    main()
