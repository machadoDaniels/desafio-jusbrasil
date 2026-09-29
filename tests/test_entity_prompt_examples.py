"""Check that synthetic demonstrations stay valid and outside the input corpus."""

import json
import re
import unittest
from importlib import import_module
from pathlib import Path

from desafio_jusbrasil.contracts import Contract
from desafio_jusbrasil.entity_extraction.entity_extraction import (
    JURISPRUDENCE_EXTRACTORS, LEGISLATION_EXTRACTORS,
)

ROOT = Path(__file__).resolve().parents[1]


def normalized(text):
    return re.sub(r"\s+", " ", text).strip().casefold()


class SyntheticPromptExamplesTests(unittest.TestCase):
    def test_examples_are_schema_valid_and_not_corpus_citations(self):
        corpus = [normalized(path.read_text()) for path in
                  (ROOT / 'desafio-jusbrasil-bracis-2026/txt').glob('*.txt')]
        self.assertTrue(corpus, 'The input corpus is required for the overlap check')
        for module in (*JURISPRUDENCE_EXTRACTORS, *LEGISLATION_EXTRACTORS):
            name = module.__name__.rsplit('.', 1)[-1].removesuffix('_extraction')
            with self.subTest(extractor=name):
                schemas = [value for value in vars(module).values()
                           if isinstance(value, type) and issubclass(value, Contract)
                           and value.__module__ == module.__name__]
                self.assertEqual(len(schemas), 1)
                schema = schemas[0]
                examples = module.EXAMPLES
                checks = json.loads((ROOT / f'tests/fixtures/entity_prompts/{name}.json').read_text())
                self.assertGreaterEqual(len(examples), 4)
                self.assertGreaterEqual(len(checks), 6)
                prompt_texts = {normalized(case['text']) for case in examples}
                check_texts = {normalized(case['text']) for case in checks}
                self.assertTrue(prompt_texts.isdisjoint(check_texts))
                for case in examples + checks:
                    self.assertEqual(set(case['expected']), set(schema.model_fields))
                    schema.model_validate(case['expected'])
                    text = normalized(case['text'])
                    self.assertFalse(any(text in document for document in corpus), case['text'])
                for case in examples:
                    self.assertIn(json.dumps(case['text'], ensure_ascii=False), module.PROMPT)
                for case in checks:
                    self.assertNotIn(json.dumps(case['text'], ensure_ascii=False), module.PROMPT)


if __name__ == '__main__':
    unittest.main()
