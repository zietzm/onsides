import argparse
import logging
import pathlib
import re

import pandas as pd
import polars as pl
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class EmaDrug(BaseModel):
    medicine_name: str  # Brand name (mostly)
    product_number: str  # EMA product number, e.g. EMEA/H/C/004913
    active_substance: str | None  # Possible multiple (e.g. `amlodipine;valsartan`)
    inn_name: str | None  # Also possible multiple (e.g. `amlodipine;valsartan`)
    atc_code: str | None  # Category, not directly mappable to RxNorm product


class DrugInfo(BaseModel):
    rxnorm_code: str | None
    concept_name: str | None


class MappedDrug(BaseModel):
    rxnorm_code: str | None
    concept_name: str | None
    n_ingredients: int


def map_drug(drug: EmaDrug, string_map: dict[str, DrugInfo]) -> MappedDrug | None:
    n_ingredients = 1 if drug.inn_name is None else drug.inn_name.count(";") + 1
    lower_name = drug.medicine_name.lower()

    # Check if medicine_name appears in the concept_name column
    maybe_drug_info = string_map.get(lower_name)
    if maybe_drug_info is not None:
        return MappedDrug(
            rxnorm_code=maybe_drug_info.rxnorm_code,
            concept_name=maybe_drug_info.concept_name,
            n_ingredients=n_ingredients,
        )

    # Check for a previous name in the medicine name, then check in concept_name
    # e.g. `clopidogrel/acetylsalicylic acid zentiva (previously duocover)`
    # -> check both `duocover` and `clopidogrel/acetylsalicylic acid zentiva`
    regex = r" \(previously (.*)\)$"
    match = re.search(regex, drug.medicine_name)
    if match is not None:
        previous_name = match.group(1)
        maybe_drug_info = string_map.get(previous_name)
        if maybe_drug_info is not None:
            return MappedDrug(
                rxnorm_code=maybe_drug_info.rxnorm_code,
                concept_name=maybe_drug_info.concept_name,
                n_ingredients=n_ingredients,
            )

        excluding_previous = re.sub(regex, "", drug.medicine_name)
        maybe_drug_info = string_map.get(excluding_previous)
        if maybe_drug_info is not None:
            return MappedDrug(
                rxnorm_code=maybe_drug_info.rxnorm_code,
                concept_name=maybe_drug_info.concept_name,
                n_ingredients=n_ingredients,
            )

    # Check the International non-proprietary name (INN) / common name
    if n_ingredients == 1 and drug.inn_name is not None:
        maybe_drug_info = string_map.get(drug.inn_name)
        if maybe_drug_info is not None:
            return MappedDrug(
                rxnorm_code=maybe_drug_info.rxnorm_code,
                concept_name=maybe_drug_info.concept_name,
                n_ingredients=n_ingredients,
            )

    return None


def deduplicate_concepts(df: pl.LazyFrame) -> pl.LazyFrame:
    return (
        df
        # Prefer RxNorm over RxNorm Extension when both are present
        .filter(
            pl.col("vocabulary_id").eq(
                pl.col("vocabulary_id").min().over(pl.col("concept_string"))
            )
        )
        # If there are still multiple rows with different values for `invalid_reason`,
        # prefer the row with the NULL value in this column (i.e. the valid one)
        .sort(pl.col("invalid_reason"))
        .with_columns(pl.first().cum_count().over("concept_string").alias("count"))
        .filter(pl.col("count").eq(pl.col("count").min().over("concept_string")))
        # If there are STILL duplicates, just take the occurring row
        .unique(subset=["concept_string"], keep="first")
    )


def load_string_map(external_data_folder: pathlib.Path) -> dict[str, DrugInfo]:
    concept_name_df = (
        pl.scan_csv(external_data_folder / "CONCEPT.csv", separator="\t", quote_char="")
        .filter(pl.col("vocabulary_id").str.contains("RxNorm"))
        .with_columns(pl.col("concept_name").str.to_lowercase().alias("concept_string"))
        .pipe(deduplicate_concepts)
        .rename({"concept_code": "rxnorm_code"})
        .select("concept_string", "concept_name", "rxnorm_code")
        .collect()
    )
    assert len(concept_name_df) == len(set(concept_name_df["concept_string"]))
    concept_synonym_df = (
        pl.scan_csv(
            external_data_folder / "CONCEPT_SYNONYM.csv", separator="\t", quote_char=""
        )
        .join(
            pl.scan_csv(
                external_data_folder / "CONCEPT_RELATIONSHIP.csv", separator="\t"
            ),
            left_on=["concept_id"],
            right_on=["concept_id_1"],
        )
        .join(
            pl.scan_csv(
                external_data_folder / "CONCEPT.csv", separator="\t", quote_char=""
            ),
            left_on=["concept_id_2"],
            right_on=["concept_id"],
        )
        .filter(
            pl.col("vocabulary_id").str.contains("RxNorm"),
            pl.col("relationship_id").eq("Maps to"),
        )
        .with_columns(
            pl.col("concept_synonym_name").str.to_lowercase().alias("concept_string")
        )
        .pipe(deduplicate_concepts)
        .rename({"concept_code": "rxnorm_code"})
        .select("concept_string", "concept_name", "rxnorm_code")
        .collect()
    )
    # Check that there are no duplicates
    assert len(concept_synonym_df) == len(set(concept_synonym_df["concept_string"]))
    merged_df = (
        pl.concat([concept_name_df, concept_synonym_df])
        .unique(subset=["concept_string"], keep="first")
        .select(
            "concept_string", pl.struct(["concept_name", "rxnorm_code"]).alias("info")
        )
    )
    assert len(merged_df) == len(set(merged_df["concept_string"]))
    return {
        d["concept_string"]: DrugInfo.model_validate(d["info"])
        for d in merged_df.to_dicts()
    }


def map_all(data_folder: pathlib.Path, external_data_folder: pathlib.Path) -> None:
    ema_df = (
        pd.read_excel(
            data_folder.joinpath("medicines-output-medicines-report_en.xlsx"),
            skiprows=8,
        )
        .pipe(pl.DataFrame)
        .filter(
            pl.col("Category") == "Human", pl.col("Medicine status").eq("Authorised")
        )
        .rename(
            {
                "Name of medicine": "medicine_name",
                "EMA product number": "product_number",
                "Active substance": "active_substance",
                "International non-proprietary name (INN) / common name": "inn_name",
                "ATC code (human)": "atc_code",
            }
        )
        .select(
            pl.col("medicine_name").str.to_lowercase(),
            "product_number",
            "active_substance",
            "inn_name",
            "atc_code",
        )
    )
    ema_drugs = [EmaDrug.model_validate(row) for row in ema_df.to_dicts()]
    string_map = load_string_map(external_data_folder)

    mapped_drugs = list()
    n_failed = 0
    for drug in ema_drugs:
        mapped_drug = map_drug(drug, string_map)
        if mapped_drug is None:
            n_failed += 1
            logger.warning(f"Could not map drug {drug}")
        else:
            mapped_drugs.append(mapped_drug)

    logger.info(
        f"Total drugs: {len(ema_drugs)}\n"
        f"Successfully mapped {len(mapped_drugs)}\n"
        f"Failed on {n_failed}\n"
    )
    pl.DataFrame(mapped_drugs).write_csv(data_folder / "ingredients.csv")


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
        "--external_data",
        type=pathlib.Path,
        required=True,
        help="Path to the where the external data is housed.",
    )
    args = parser.parse_args()
    map_all(args.data_folder, args.external_data)


if __name__ == "__main__":
    main()
