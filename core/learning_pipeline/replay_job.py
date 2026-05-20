from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

from core.learning_pipeline.dataset_exporter import LearningDatasetExporter

logger = logging.getLogger(__name__)


@dataclass
class ReplayResult:
    output_dir: str
    summary_file: str
    curated_file: str
    total_records: int
    repeated_issues: int
    route_issues: int
    answer_issues: int


class LearningReplayJob:
    def __init__(self) -> None:
        self.exporter = LearningDatasetExporter()

    @staticmethod
    def _group_key(record: Dict[str, object]) -> str:
        normalized = str(record.get("normalized_query") or "").strip()
        if normalized:
            return normalized
        return str(record.get("input") or "").strip().lower()

    def _build_summary(self, records: List[Dict[str, object]]) -> Dict[str, object]:
        grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
        for record in records:
            grouped[self._group_key(record)].append(record)

        repeated_groups = []
        for key, items in grouped.items():
            if len(items) < 2:
                continue
            task_type = items[0].get("task_type", "unknown")
            outputs = Counter(str(item.get("output") or "") for item in items if item.get("output"))
            repeated_groups.append(
                {
                    "group_key": key,
                    "task_type": task_type,
                    "count": len(items),
                    "top_output": outputs.most_common(1)[0][0] if outputs else "",
                    "sample_input": items[0].get("input", ""),
                }
            )

        repeated_groups.sort(key=lambda item: item["count"], reverse=True)
        return {
            "total_groups": len(grouped),
            "repeated_groups": repeated_groups,
        }

    def _build_curated_records(self, records: List[Dict[str, object]]) -> List[Dict[str, object]]:
        curated: List[Dict[str, object]] = []
        for record in records:
            task_type = record.get("task_type")
            rating = int(record.get("rating") or 0)
            feedback_type = str(record.get("feedback_type") or "")

            if task_type == "answer":
                if rating >= 4 or feedback_type in {"helpful", "positive", "good_answer"}:
                    curated.append(record)
                elif record.get("corrected_answer"):
                    curated.append(record)
            elif task_type == "route":
                curated.append(record)
        return curated

    def run(
        self,
        output_dir: str = "learning_replay",
        answer_memory: object = None,
    ) -> ReplayResult:
        records = self.exporter.build_dataset()
        summary = self._build_summary(records)
        curated = self._build_curated_records(records)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        summary_file = output_path / "replay_summary.json"
        curated_file = output_path / "replay_curated.jsonl"

        with summary_file.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=True, default=str)

        with curated_file.open("w", encoding="utf-8") as handle:
            for record in curated:
                handle.write(json.dumps(record, ensure_ascii=True, default=str) + "\n")

        if answer_memory and hasattr(answer_memory, "refresh"):
            try:
                answer_memory.refresh()
            except Exception as exc:
                logger.warning("[LearningReplay] Answer memory refresh failed: %s", exc)

        route_count = len([r for r in records if r.get("task_type") == "route"])
        answer_count = len([r for r in records if r.get("task_type") == "answer"])
        repeated_count = len(summary.get("repeated_groups", []))

        logger.info(
            "[LearningReplay] Built replay bundle with %s records (%s answer, %s route)",
            len(records),
            answer_count,
            route_count,
        )

        return ReplayResult(
            output_dir=str(output_path),
            summary_file=str(summary_file),
            curated_file=str(curated_file),
            total_records=len(records),
            repeated_issues=repeated_count,
            route_issues=route_count,
            answer_issues=answer_count,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    job = LearningReplayJob()
    result = job.run()
    print(json.dumps(asdict(result), indent=2))
