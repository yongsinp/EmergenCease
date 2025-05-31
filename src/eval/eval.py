from src.extraction.llm import Extractor


class Evaluator:
    def __init__(self, model: str):
        self._extractor = Extractor(model)

    def evaluate(self, input_text, gold_standard, columns, metrics):
        ...