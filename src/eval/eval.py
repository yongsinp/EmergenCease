import difflib
from dataclasses import dataclass
from functools import wraps
from typing import Iterable

import Levenshtein as levenshtein
import nltk
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from tqdm import tqdm

from src.extraction.llm import Extractor
from src.utils.paths import DATA_DIR


@dataclass
class Result:
    event: float  # F1
    location_em: float  # Set F1 (exact match)
    location_partial: float  # Set F1 (partial match)
    sender: float  # Exact match
    time_em: float  # Exact match
    time_rouge_l: float  # ROUGE-L
    url_em: float  # Set F1 (exact match)
    url_partial: float  # Set F1 (partial match)
    failed: int


class Evaluator:
    def __init__(self, model: str):
        self._extractor = Extractor(model)
        self._rouge = None

    @staticmethod
    def _normalize(input: str) -> str:
        """Normalizes the input string by stripping whitespace and converting to lowercase."""
        return str(input).strip().lower()

    @staticmethod
    def _normalize_input(func):
        @wraps(func)
        def wrapper(self, pred, ref, *args, **kwargs):
            pred = self._normalize(pred)
            ref = self._normalize(ref)
            return func(self, pred, ref, *args, **kwargs)

        return wrapper

    @_normalize_input
    def exact_match(self, pred: str, ref: str) -> int:
        return int(pred == ref)

    @_normalize_input
    def partial_match(self, pred: str, ref: str) -> int:
        """Measures partial correctness by checking if the answer is a substring of the prediction."""
        return int(pred in ref)

    @_normalize_input
    def fuzzy_match(self, pred: str, ref: str, method: str = "Gestalt") -> float:
        """
        Measures fuzzy match using SequenceMatcher to calculate the ratio of similarity.
        Returns a float between 0 and 1, where 1 means exact match.
        """
        match method:
            case "Gestalt":
                return difflib.SequenceMatcher(None, pred, ref).ratio()
            case "Levenshtein":
                return levenshtein.ratio(pred, ref)
            case _:
                raise ValueError(f"Unknown fuzzy match method: {method}")

    @_normalize_input
    def f1(self, pred: str, ref: str) -> float:
        """Calculates the F1 score between the predicted and actual answer."""
        raise NotImplementedError()

    @_normalize_input
    def token_f1(self, pred: str, ref: str, tokenization: str = "space") -> float:
        """
        Calculates the F1 score between the predicted and actual answer based on tokenization.

        Parameters:
            pred: The predicted answer.
            ref: The actual answer.
            tokenization: The method of tokenization to use. ["space", "nltk"]
        """
        match tokenization:
            case "space":
                pred_tokens = pred.split()
                ref_tokens = ref.split()
            case "nltk":
                pred_tokens = nltk.word_tokenize(pred)
                ref_tokens = nltk.word_tokenize(ref)

        return self.set_f1(pred_tokens, ref_tokens)

    def set_f1(self, pred: Iterable[str], ref: Iterable[str], match_: str = "exact", ratio: float = 0.7) -> float:
        """
        Calculates the F1 score between the predicted and actual answer.

        Parameters:
            pred: Predicted answers as a list.
            ref: Actual answers as a list.
            match_: The type of match to use for F1 calculation. ["exact", "partial"]
            ratio: The threshold for fuzzy matching, only used in "partial" match.
        """
        # Explicitly handle empty arguments
        if not pred and not ref:
            return 1.0
        elif not isinstance(pred, Iterable) or not isinstance(ref, Iterable) or not pred or not ref:
            return 0.0

        pred = set(self._normalize(p) for p in pred)
        ref = set(self._normalize(r) for r in ref)

        match match_:
            case "exact":
                common = len(pred.intersection(ref))
            case "partial":
                common = sum(1 for p in pred if any(self.fuzzy_match(p, r, "Gestalt") > ratio for r in ref))
            case _:
                raise ValueError(f"Unknown match type: {match_}")

        if common == 0:
            return 0.0

        precision = common / len(pred)
        recall = common / len(ref)

        return (2 * precision * recall) / (precision + recall)

    @_normalize_input
    def bleu(self, pred: str, ref: str) -> float:
        """Calculates the BLEU score between the predicted and actual answer."""
        # Tokenize using nltk
        pred_tokens = nltk.word_tokenize(pred)
        ans_tokens = nltk.word_tokenize(ref)

        return sentence_bleu([ans_tokens], pred_tokens)

    @_normalize_input
    def rouge_l(self, pred: str, ref: str) -> float:
        """Calculates the ROUGE score between the predicted and actual answer."""
        if self._rouge is None:
            self._rouge = Rouge()

        # Explicitly handle empty strings
        if not pred and not ref:
            return  1.0
        elif not pred or not ref:
            return 0.0

        scores = self._rouge.get_scores(pred, ref)
        return scores[0]['rouge-l']['f']

    @_normalize_input
    def rouge_s(self, pred: str, ref: str) -> float:
        """
        Calculates the ROUGE-S score between the predicted and actual answer.
        This may be a better fit for time extraction tasks.
        """
        raise NotImplementedError()

    def evaluate(self, eval_file: str) -> Result:
        data = pd.read_csv(eval_file)
        data = data.fillna("").replace({None: ""})  # Replace NaNs and Nones with empty strings

        # Initialize counters
        event = 0
        location_em = 0
        location_partial = 0
        sender = 0
        time_em = 0
        time_rouge_l = 0
        url_em = 0
        url_partial = 0
        failed = 0

        for index, row in tqdm(data.iterrows(), total=len(data), desc="Evaluating"):
            input = {
                "headline": row["headline"],
                "description": row["description"],
                "instruction": row["instruction"]
            }
            try:
                extracted_data = self._extractor.extract(**input)

                # Todo: bidirectional match
                # Todo: both partial and exact match
                event += self.exact_match(extracted_data['event'], row['event'])
                location_em += self.set_f1(extracted_data['location'].split(";"), row['location'].split(";"),
                                           match_="exact")
                location_partial += self.set_f1(extracted_data['location'].split(";"), row['location'].split(";"),
                                                match_="partial")
                sender += self.exact_match(extracted_data['sender'], row['sending_agency'])
                time_em += self.exact_match(extracted_data['expires'], row['time'])
                time_rouge_l += self.rouge_l(extracted_data['expires'], row['time'])
                url_em += self.set_f1(extracted_data['url'].split(";"), row['url'].split(";"), match_="exact")
                url_partial += self.set_f1(extracted_data['url'].split(";"), row['url'].split(";"), match_="partial")
            except RuntimeError as e:
                failed += 1
                pass

        len_data = len(data) - failed
        return Result(
            event=event / len_data,
            location_em=location_em / len_data,
            location_partial=location_partial / len_data,
            sender=sender / len_data,
            time_em=time_em / len_data,
            time_rouge_l=time_rouge_l / len_data,
            url_em=url_em / len_data,
            url_partial=url_partial / len_data,
            failed=failed
        )


if __name__ == '__main__':
    models = [
        "unsloth/Llama-3.2-3B",
        "unsloth/Llama-3.2-3B-Instruct",
        "meta-llama/Meta-Llama-3.1-8B-Instruct"
    ]
    evalulator = Evaluator(models[1])
    result = evalulator.evaluate(DATA_DIR / "finetune" / "finetune_val.csv")
    print(result)
