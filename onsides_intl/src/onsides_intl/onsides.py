from pathlib import Path

from pydantic import BaseModel

from onsides_intl.predict import TextSettings, predict
from onsides_intl.stringsearch import (
    ContextSettings,
    IndexedText,
    parse_texts,
)


class DrugAdverseEffect(BaseModel):
    label_id: int
    adverse_effect_id: int
    prediction: float


class BertModelConfig(BaseModel):
    network_path: Path
    weights_path: Path
    batch_size: int | None = None
    device_id: int | None = None
    text_settings: TextSettings | None = None


def extract_adverse_effects(
    drug_labels: list[IndexedText],
    adverse_effects: list[IndexedText],
    bert_model_config: BertModelConfig,
    match_threshold: float = 0.4633,
    string_match_settings: ContextSettings | None = None,
) -> list[DrugAdverseEffect]:
    # Find all string matches of adverse effects in the labels
    matches = parse_texts(
        drug_labels, adverse_effects, string_match_settings, progress=True
    )

    # Predict the probability of each match being a true adverse effect
    context_objs = [IndexedText(text_id=x.match_id, text=x.context) for x in matches]
    predictions = predict(
        context_objs,
        network_path=bert_model_config.network_path,
        weights_path=bert_model_config.weights_path,
        text_settings=bert_model_config.text_settings,
        batch_size=bert_model_config.batch_size,
        device_id=bert_model_config.device_id,
    )

    # Filter for matches with a prediction above the threshold
    match_id_to_match = {x.match_id: x for x in matches}
    filtered_matches = list()
    for prediction in predictions:
        if prediction.prediction >= match_threshold:
            match = match_id_to_match[prediction.text_id]
            result = DrugAdverseEffect(
                label_id=match.text_id,
                adverse_effect_id=match.term_id,
                prediction=prediction.prediction,
            )
            filtered_matches.append(result)
    return filtered_matches
