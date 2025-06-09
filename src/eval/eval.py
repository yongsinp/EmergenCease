import argparse
import difflib
import json
import logging
import os.path
import time
from dataclasses import dataclass
from functools import wraps
from typing import Iterable, Union

import Levenshtein as levenshtein
import nltk
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from tqdm import tqdm

from src.extraction.llm import Extractor, get_logger
from src.utils.paths import DATA_DIR, MODEL_DIR


@dataclass
class Result:
    """A dataclass to store evaluation results."""
    event: float  # F1
    location_em: float  # Set F1 (exact match)
    location_partial: float  # Set F1 (partial match)
    sender: float  # Exact match
    time_em: float  # Exact match
    time_rouge_l: float  # ROUGE-L
    url_em: float  # Set F1 (exact match)
    url_partial: float  # Set F1 (partial match)
    failed: int
    time_per_sample: int


class Evaluator:
    """A class to evaluate LLM extractor performance."""
    _logger = None

    def __init__(self, model: str, adapter: str = None) -> None:
        """
        Initializes the Translator with a specified LLM.

        Parameters:
            model: A local path or Hugging Face model identifier for the model used in extraction.
            adapter: Optional path to a model adapter.
        """
        self._initialize_class_attributes()

        self._extractor = Extractor(model, adapter=adapter)
        self._rouge = None

    @staticmethod
    def _normalize(input: str) -> str:
        """Normalizes the input string by stripping whitespace and converting to lowercase."""
        return str(input).strip().lower()

    @staticmethod
    def _normalize_input(func):
        """Decorator to normalize input strings before passing them to the evaluation functions."""

        @wraps(func)
        def wrapper(self, pred, ref, *args, **kwargs):
            pred = self._normalize(pred)
            ref = self._normalize(ref)
            return func(self, pred, ref, *args, **kwargs)

        return wrapper

    @classmethod
    def _initialize_class_attributes(cls) -> None:
        """Initializes class-level attributes."""
        if cls._logger is None:
            cls._logger = logging.getLogger(cls.__name__)
            cls._logger.setLevel(logging.INFO)

    @classmethod
    def set_logger_level(cls, level: Union[str, int] = logging.INFO) -> None:
        """
        Sets logging level for the class logger.

        Parameters:
            level: Logging level. [DEBUG, INFO, WARNING, ERROR, CRITICAL]
        """
        cls._logger.setLevel(level)

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
        Measures the fuzzy match between the predicted and actual answer using specified method.

        Parameters:
            pred: The predicted answer.
            ref: The actual answer.
            method: Fuzzy matching method to use. ["Gestalt", "Levenshtein"]

        Returns:
            A float representing the fuzzy match score between 0 and 1.
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

        Returns:
            A float representing the F1 score between 0 and 1.
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
            pred: A list of predicted answers.
            ref: A list of actual answers.
            match_: Match type to use. "partial" uses "Gestalt" fuzzy matching. ["exact", "partial"]
            ratio: Threshold for fuzzy matching, only used for "partial" match.

        Returns:
            A float representing the F1 score between 0 and 1.
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
        """
        Calculates the BLEU score between the predicted and actual answer.

        Parameters:
            pred: The predicted answer.
            ref: The actual answer.

        Returns:
            A float representing the BLEU score between 0 and 1.
        """
        # Tokenize using nltk
        pred_tokens = nltk.word_tokenize(pred)
        ans_tokens = nltk.word_tokenize(ref)

        return sentence_bleu([ans_tokens], pred_tokens)

    @_normalize_input
    def rouge_l(self, pred: str, ref: str) -> float:
        """
        Calculates the ROUGE score between the predicted and actual answer.

        Parameters:
            pred: The predicted answer.
            ref: The actual answer.

        Returns:
            A float representing the ROUGE-L score between 0 and 1.
        """
        if self._rouge is None:
            self._rouge = Rouge()

        # Explicitly handle empty strings
        if not pred and not ref:
            return 1.0
        elif not pred or not ref:
            return 0.0

        scores = self._rouge.get_scores(pred, ref)
        return scores[0]['rouge-l']['f']

    @_normalize_input
    def rouge_s(self, pred: str, ref: str) -> float:
        """
        Calculates the ROUGE-S score between the predicted and actual answer.

        This may be a better fit for time extraction tasks.
        This method has not been implemented yet.

        Parameters:
            pred: The predicted answer.
            ref: The actual answer.

        Returns:
            A float representing the ROUGE-S score between 0 and 1.
        """
        raise NotImplementedError()

    def evaluate(self, eval_file: str) -> Result:
        """
        Evaluates the extractor's performance on a given evaluation file.

        Parameters:
            eval_file: Path to the CSV file containing evaluation data.

        Returns:
            Result: A Result instance containing evaluation metrics.
        """
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

        start = time.time()  # For processing time measurement
        for index, row in tqdm(data.iterrows(), total=len(data), desc="Evaluating"):
            input = {
                "headline": row["headline"],
                "description": row["description"],
                "instruction": row["instruction"]
            }
            self._logger.debug(f"\nInput: {json.dumps(input, indent=4)}")

            try:
                extracted_data = self._extractor.extract(**input)
                self._logger.debug(f"\nExtracted data: {json.dumps(extracted_data, indent=4)}")

                # Todo: Try bidirectional match
                # Todo: Simplify evaluation by putting these in a dictionary
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
                self._logger.warning(f"Error processing {row['uuid']}: {e}")
                failed += 1

        # Adjust for failed samples
        len_data = len(data) - failed

        result = Result(
            event=event / len_data,
            location_em=location_em / len_data,
            location_partial=location_partial / len_data,
            sender=sender / len_data,
            time_em=time_em / len_data,
            time_rouge_l=time_rouge_l / len_data,
            url_em=url_em / len_data,
            url_partial=url_partial / len_data,
            failed=failed,
            time_per_sample=(time.time() - start) / len(data)
        )
        self._logger.info(result)

        return result


def main():
    """Example code for the Evaluator."""
    parser = argparse.ArgumentParser(description="Script for evaluating LLM extractors.")

    # Evaluation arguments
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.2-1B-Instruct',
                        help='Model to use for extraction (default: meta-llama/Llama-3.2-1B-Instruct)')
    parser.add_argument('--adapter', type=str, default=None,
                        help='Path to the model adapter (default: None)')
    parser.add_argument('--test-data', type=str, default=None,
                        help='Path to the test data CSV file. `/data/finetune/finetune_test.csv` is used if not provided. (default: None)')
    parser.add_argument('--runs', type=int, default=5,
                        help='Number of runs for averaging results (default: 5)')

    # Logger arguments
    parser.add_argument('--log-level', type=str, default='INFO',
                        help='Logging level [DEBUG, INFO, WARNING, ERROR, CRITICAL] (default: INFO)')

    args = parser.parse_args()

    data_path = args.test_data if args.test_data else os.path.join(DATA_DIR, "finetune", "finetune_test.csv")
    evalulator = Evaluator(args.model, adapter=args.adapter)
    evalulator.set_logger_level(args.log_level)

    # Run multiple evaluations to get more reliable results
    runs = args.runs
    results = []
    for _ in range(runs):
        result = evalulator.evaluate(data_path)
        results.append(result)

    # Todo: Simplify
    # Print average
    average_result = Result(
        event=sum(r.event for r in results) / runs,
        location_em=sum(r.location_em for r in results) / runs,
        location_partial=sum(r.location_partial for r in results) / runs,
        sender=sum(r.sender for r in results) / runs,
        time_em=sum(r.time_em for r in results) / runs,
        time_rouge_l=sum(r.time_rouge_l for r in results) / runs,
        url_em=sum(r.url_em for r in results) / runs,
        url_partial=sum(r.url_partial for r in results) / runs,
        failed=sum(r.failed for r in results) / runs,
        time_per_sample=sum(r.time_per_sample for r in results) / runs
    )
    print(f"\nAveraged {average_result}")


if __name__ == '__main__':
    main()
