"""Run from the repo root: python -m unittest tests.test_instrumentation"""

import os
import tempfile
import threading
import unittest

from core.instrumentation import Trace, current_trace, load_events, replay


class TraceTest(unittest.TestCase):
    def test_usage_is_attributed_to_all_open_stages(self):
        with Trace() as trace:
            with trace.stage("outer"):
                current_trace().record_llm_call("m", 10, 2, 0.5, 0.1)
                with trace.stage("inner"):
                    current_trace().record_llm_call("m", 5, 1, 0.25, 0.1)
        ends = {event["stage"]: event for event in trace.events if event["type"] == "stage_end"}
        self.assertEqual(
            (ends["inner"]["llm_calls"], ends["inner"]["input_tokens"], ends["inner"]["cost_usd"]), (1, 5, 0.25)
        )
        self.assertEqual(
            (ends["outer"]["llm_calls"], ends["outer"]["input_tokens"], ends["outer"]["cost_usd"]), (2, 15, 0.75)
        )
        self.assertEqual(trace.usage["output_tokens"], 3)

    def test_events_from_threads_are_ordered_and_tagged_with_the_stage(self):
        seen = []
        with Trace(on_event=seen.append) as trace:
            with trace.stage("work"):
                threads = [threading.Thread(target=trace.emit, args=("tick",), kwargs={"i": i}) for i in range(50)]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()
        self.assertEqual([event["seq"] for event in seen], list(range(len(seen))))
        self.assertTrue(all(event["stage"] == "work" for event in seen if event["type"] == "tick"))

    def test_no_trace_outside_of_a_trace(self):
        self.assertFalse(current_trace().enabled)
        with current_trace().stage("ignored"):
            current_trace().emit("ignored")
        with Trace() as trace:
            self.assertIs(current_trace(), trace)
        self.assertFalse(current_trace().enabled)

    def test_save_and_replay(self):
        with Trace() as trace:
            trace.emit("a", value=1)
            trace.emit("b", value="ü")
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "trace.json")
            trace.save(path)
            events = load_events(path)
        self.assertEqual(list(replay(events, speed=1000)), trace.events)


if __name__ == "__main__":
    unittest.main()
