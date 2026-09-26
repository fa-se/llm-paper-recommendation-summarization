"""Run from the repo root: uv run python -m unittest tests.test_tasks"""

import json
import unittest

from core.config import Settings
from core.llm_interfaces.openai import completion_params
from core.llm_interfaces.tasks import SetwiseComparisonTask, TailoredSummaryTask


class SetwiseComparisonTaskTest(unittest.TestCase):
    def test_prompt_labels_the_passages(self):
        system, user = SetwiseComparisonTask("q", ["first", "second", "third"]).messages()
        self.assertEqual(system.role, "system")
        self.assertIn('Given a query "q"', user.content)
        self.assertIn('Passage A: "first"\n\nPassage B: "second"\n\nPassage C: "third"', user.content)

    def test_parse_response(self):
        task = SetwiseComparisonTask("q", ["a", "b", "c"])
        for response, expected in [("A", 0), ("Passage B", 1), ("C\n", 2), ("The answer is Passage C.", 2)]:
            self.assertEqual(task.parse_response(response), expected, response)
        for response in ["D", "Passage D", "b", "none"]:
            self.assertIsNone(task.parse_response(response), response)


class TailoredSummaryTaskTest(unittest.TestCase):
    def test_prompt_contains_the_interests_and_the_abstract(self):
        _, user = TailoredSummaryTask("my interests", "the abstract").messages()
        self.assertIn("## Research Interest Description\n        my interests", user.content)
        self.assertIn("## Abstract\n        the abstract", user.content)
        self.assertIn('"FINAL_ANSWER": ""', user.content)

    def test_schema_requires_every_step_in_order(self):
        schema = TailoredSummaryTask("i", "a").response_format()["json_schema"]["schema"]
        steps = schema["properties"]["Reasoning Structure"]
        self.assertEqual(steps["required"][-1], "FINAL_ANSWER")
        self.assertEqual(steps["required"][0], "Step 1: Identify Key Findings")
        self.assertFalse(steps["additionalProperties"])

    def test_parse_response(self):
        response = json.dumps({"Reasoning Structure": {"FINAL_ANSWER": "summary"}})
        self.assertEqual(TailoredSummaryTask("i", "a").parse_response(response), {"FINAL_ANSWER": "summary"})


class ModelConfigTest(unittest.TestCase):
    def test_defaults_and_overrides(self):
        models = Settings.from_env({}).models
        self.assertEqual((models["quality"].name, models["quality"].reasoning_effort), ("gpt-6-sol", "low"))
        self.assertEqual((models["rerank"].name, models["rerank"].reasoning_effort), ("gpt-6-luna", "none"))
        env = {"OPENAI_RERANK_MODEL": "gpt-4o-mini-2024-07-18", "OPENAI_RERANK_REASONING_EFFORT": ""}
        models = Settings.from_env(env).models
        self.assertEqual((models["rerank"].name, models["rerank"].reasoning_effort), ("gpt-4o-mini-2024-07-18", None))

    def test_completion_params(self):
        # reasoning models: temperature only with effort "none"; older models: no reasoning_effort
        self.assertEqual(completion_params("gpt-6-luna", "none", 0.0), {"reasoning_effort": "none", "temperature": 0.0})
        self.assertEqual(completion_params("gpt-6-sol", "low", 0.0), {"reasoning_effort": "low"})
        self.assertEqual(completion_params("gpt-4o-mini-2024-07-18", "none", 0.0), {"temperature": 0.0})

    def test_database_url_escapes_the_password(self):
        env = {"DB_USER": "u", "DB_PASSWORD": "p@ss/word", "DB_HOST": "h", "DB_NAME": "d"}
        url = Settings.from_env(env).database_url
        self.assertEqual(url.password, "p@ss/word")
        self.assertEqual(url.host, "h")
        self.assertEqual(url.query["options"], "-csearch_path=public,bm_catalog")


if __name__ == "__main__":
    unittest.main()
