import tqdm.auto as tqdm
from ahocorasick import Automaton
from pydantic import BaseModel


class SearchTerm(BaseModel):
    term_id: int
    term: str


class IndexedText(BaseModel):
    text_id: int
    text: str


class MatchContext(BaseModel):
    text_id: int
    term_id: int
    term: str
    context: str


class ContextSettings(BaseModel):
    nwords: int = 125
    prop_before: float = 0.125


def parse_texts(
    texts: list[IndexedText],
    terms: list[SearchTerm],
    context_settings: ContextSettings,
    progress: bool = False,
) -> list[MatchContext]:
    """
    Find all term occurrences in a series of text strings, returning each match
    with the surrounding context for use with BERT.
    """
    term_tree = _build_search_tree(terms)
    matches = list()
    texts_to_iter = tqdm.tqdm(texts) if progress else texts
    for text_obj in texts_to_iter:
        found_terms = _find_terms_in_text(text_obj.text, term_tree)
        for found_term in found_terms:
            match = MatchContext(
                text_id=text_obj.text_id,
                term_id=found_term.term_id,
                term=found_term.term,
                context=_build_bert_string(
                    text_obj.text,
                    found_term,
                    nwords=context_settings.nwords,
                    prop_before=context_settings.prop_before,
                ),
            )
            matches.append(match)
    return matches


class _FoundTerm(BaseModel):
    term_id: int
    term: str
    start: int
    end: int


def _build_search_tree(terms: list[SearchTerm]) -> Automaton:
    """
    Builds an Aho-Corasick tree from a list of terms.
    """
    tree = Automaton(str, str)
    for obj in terms:
        tree.add_word(obj.term, obj.model_dump_json())
    tree.make_automaton()
    return tree


def _find_terms_in_text(text: str, term_tree: Automaton) -> list[_FoundTerm]:
    """
    Finds all terms in a text using an Aho-Corasick tree.
    """
    found_terms = list()
    for end_index, obj_json in term_tree.iter(text):
        obj = SearchTerm.model_validate_json(obj_json)
        start_index = end_index - len(obj.term) + 1
        obj = _FoundTerm(
            term_id=obj.term_id,
            term=obj.term,
            start=start_index,
            end=end_index,
        )
        found_terms.append(obj)
    return found_terms


def _build_bert_string(
    text: str,
    match: _FoundTerm,
    nwords: int = 125,
    prop_before: float = 0.125,
) -> str:
    """
    Extract surrounding context around a text match for use with BERT.
    """
    term_nwords = len(match.term.split())
    n_words_before = prop_before * (nwords - 2 * term_nwords)
    n_words_after = (1 - prop_before) * (nwords - 2 * term_nwords)
    n_words_before = max(int(n_words_before), 1)
    n_words_after = max(int(n_words_after), 1)
    before_words = text[: match.start].split()[-n_words_before:]
    after_words = text[match.end :].split()[:n_words_after]
    words_list = [match.term] + before_words + ["EVENT"] + after_words
    result = " ".join(words_list)
    return result
