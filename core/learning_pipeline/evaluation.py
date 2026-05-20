from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    total_records: int
    answer_records: int
    routing_records: int
    positive_feedback: int
    negative_feedback: int
    repeated_queries: int
    coverage_by_task: Dict[str, int]


class LearningEvaluator:
    def load_jsonl(self, path: str) -> List[Dict]:
        file_path = Path(path)
        if not file_path.exists():
            return []
        records: List[Dict] = []
        with file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except Exception:
                    continue
        return records

    def evaluate(self, dataset_path: str) -> EvaluationResult:
        records = self.load_jsonl(dataset_path)
        task_counter = Counter()
        query_counter = Counter()
        positive = 0
        negative = 0

        for record in records:
            task = str(record.get("task_type") or "unknown")
            task_counter[task] += 1
            query_key = str(record.get("normalized_query") or record.get("input") or "").strip().lower()
            if query_key:
                query_counter[query_key] += 1

            rating = int(record.get("rating") or 0)
            feedback_type = str(record.get("feedback_type") or "")
            if rating >= 4 or feedback_type in {"helpful", "positive", "good_answer"}:
                positive += 1
            if rating <= 2 or feedback_type in {"bad_answer", "wrong_bot", "needs_more_detail", "not_context_aware"}:
                negative += 1

        repeated = sum(1 for count in query_counter.values() if count > 1)
        return EvaluationResult(
            total_records=len(records),
            answer_records=task_counter.get("answer", 0),
            routing_records=task_counter.get("route", 0),
            positive_feedback=positive,
            negative_feedback=negative,
            repeated_queries=repeated,
            coverage_by_task=dict(task_counter),
        )


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:
        print("Usage: python -m core.learning_pipeline.evaluation <dataset.jsonl>")
        raise SystemExit(1)
    evaluator = LearningEvaluator()
    result = evaluator.evaluate(sys.argv[1])
    print(json.dumps(asdict(result), indent=2))
