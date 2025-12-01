"""RAGAS evaluation runner for HR Chatbot."""

import json
import os
import time
import logging
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, tzinfo

import httpx
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy

from .config import get_ragas_llm, get_ragas_embeddings
from .retrieval_metrics import calculate_source_metrics, calculate_aggregate_metrics
from .dataset_validator import validate_dataset

logger = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    """Evaluation configuration."""

    api_base_url: str = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")
    dataset_path: str = "datasets/eval_dataset.json"
    output_dir: str = "results"
    timeout_seconds: int = 120


class EvaluationRunner:
    """Runs RAGAS evaluation against the HR Chatbot API."""

    def __init__(self, config: EvalConfig):
        self.config = config
        self.client = httpx.Client(timeout=config.timeout_seconds)
        self.llm = get_ragas_llm()
        self.embeddings = get_ragas_embeddings()

    def load_dataset(self) -> list[dict]:
        """Load evaluation dataset from JSON file."""
        dataset_path = Path(self.config.dataset_path)
        if not dataset_path.is_absolute():
            # Resolve relative paths from package directory (evaluation/)
            package_dir = Path(__file__).parent.parent.parent
            dataset_path = package_dir / dataset_path

        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {dataset_path}\n"
                "Create evaluation dataset in a separate session."
            )

        with open(dataset_path) as f:
            data = json.load(f)

        logger.info(f"Loaded {len(data['samples'])} evaluation samples")
        return data["samples"]

    def call_chat_api(self, question: str) -> dict:
        """Call HR Chatbot API and return response with contexts."""
        response = self.client.post(
            f"{self.config.api_base_url}/chat",
            json={"message": question}
        )
        response.raise_for_status()
        return response.json()

    def collect_responses(self, samples: list[dict]) -> tuple[dict, list[dict]]:
        """
        Call API for each sample and collect responses.

        Returns tuple of:
        1. RAGAS-expected format dict:
           - question: list of questions
           - answer: list of generated answers
           - ground_truth: list of expected answers (if available)
        2. List of retrieval metrics per sample
        """
        questions = []
        answers = []
        ground_truths = []
        retrieval_metrics = []

        for i, sample in enumerate(samples):
            logger.info(
                f"Processing sample {i+1}/{len(samples)}: "
                f"{sample['question'][:50]}..."
            )

            try:
                api_response = self.call_chat_api(sample["question"])

                questions.append(sample["question"])
                answers.append(api_response["response"])
                ground_truths.append(sample.get("ground_truth", ""))

                # Calculate retrieval metrics (compare URLs)
                retrieved_urls = [src["url"] for src in api_response.get("sources", [])]
                expected_urls = sample.get("expected_sources", [])
                retrieval_result = calculate_source_metrics(retrieved_urls, expected_urls)
                retrieval_metrics.append({
                    "question": sample["question"],
                    **retrieval_result
                })

            except Exception as e:
                logger.error(f"Failed to process sample: {e}")
                questions.append(sample["question"])
                answers.append(f"ERROR: {str(e)}")
                ground_truths.append(sample.get("ground_truth", ""))

                # Add failed retrieval metrics
                retrieval_metrics.append({
                    "question": sample["question"],
                    "precision": 0.0,
                    "recall": 0.0,
                    "true_positives": 0,
                    "false_positives": 0,
                    "false_negatives": len(sample.get("expected_sources", [])),
                })

        ragas_data = {
            "question": questions,
            "answer": answers,
            "ground_truth": ground_truths,
        }

        return ragas_data, retrieval_metrics

    def run(self) -> dict:
        """Run full evaluation and return results."""
        logger.info("Starting dual-purpose evaluation (retrieval + generation)")
        start_time = time.perf_counter()

        samples = self.load_dataset()

        # Validate dataset before evaluation
        logger.info("Validating dataset...")
        validation_errors = validate_dataset(samples)
        if validation_errors:
            error_msg = "\n".join(f"  - {error}" for error in validation_errors)
            logger.error(f"Dataset validation failed:\n{error_msg}")
            raise ValueError(
                f"Invalid dataset ({len(validation_errors)} errors):\n{error_msg}"
            )

        logger.info("Collecting API responses...")
        ragas_data, retrieval_metrics = self.collect_responses(samples)

        # Convert to HuggingFace Dataset (RAGAS requirement)
        dataset = Dataset.from_dict(ragas_data)

        # Select metrics (only answer_relevancy, others require contexts)
        metrics = [answer_relevancy]

        logger.info(f"Running RAGAS with metrics: {[m.name for m in metrics]}")

        ragas_results = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=self.llm,
            embeddings=self.embeddings,
        )

        elapsed = time.perf_counter() - start_time

        # Calculate aggregate retrieval metrics
        aggregate_retrieval = calculate_aggregate_metrics(retrieval_metrics)

        # Build comprehensive output
        ragas_df = ragas_results.to_pandas()
        output = {
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": round(elapsed, 2),
            "num_samples": len(samples),
            "retrieval_metrics": {
                "avg_source_precision": aggregate_retrieval["avg_source_precision"],
                "avg_source_recall": aggregate_retrieval["avg_source_recall"],
            },
            "generation_metrics": {
                metric.name: ragas_results[metric.name]
                for metric in metrics
            },
            "per_sample": [
                {
                    "question": ragas_df.iloc[idx]["user_input"],
                    "answer": ragas_df.iloc[idx]["response"],
                    "retrieval": {
                        "precision": retrieval_metrics[idx]["precision"],
                        "recall": retrieval_metrics[idx]["recall"],
                    },
                    "generation": {
                        metric.name: float(ragas_df.iloc[idx][metric.name])
                        for metric in metrics
                    },
                }
                for idx in range(len(samples))
            ],
        }

        logger.info(f"Evaluation complete in {elapsed:.1f}s")
        logger.info(
            f"Retrieval: P={aggregate_retrieval['avg_source_precision']:.2f}, "
            f"R={aggregate_retrieval['avg_source_recall']:.2f}, "
        )
        logger.info(f"Generation: {output['generation_metrics']}")

        return output


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="HR Chatbot RAGAS Evaluation")
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000/api/v1",
        help="API base URL"
    )
    parser.add_argument(
        "--dataset",
        default="datasets/eval_dataset.json",
        help="Path to evaluation dataset"
    )
    parser.add_argument(
        "--output",
        default="results",
        help="Output directory for results"
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    config = EvalConfig(
        api_base_url=args.api_url,
        dataset_path=args.dataset,
        output_dir=args.output,
    )

    runner = EvaluationRunner(config)
    results = runner.run()

    # Save results
    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        # Resolve relative paths from package directory (evaluation/)
        package_dir = Path(__file__).parent.parent.parent
        output_dir = package_dir / output_dir

    output_path = output_dir / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Samples Evaluated: {results['num_samples']}")
    print(f"Duration: {results['duration_seconds']:.1f}s")
    print()

    print("RETRIEVAL METRICS (Are we finding the right docs?)")
    ret_metrics = results["retrieval_metrics"]
    print(f"  Source Precision: {ret_metrics['avg_source_precision']:.2f} "
          f"({ret_metrics['avg_source_precision']*100:.0f}% of retrieved docs are correct)")
    print(f"  Source Recall:    {ret_metrics['avg_source_recall']:.2f} "
          f"({ret_metrics['avg_source_recall']*100:.0f}% of expected docs were found)")
    print()

    print("GENERATION METRICS (Answer quality)")
    gen_metrics = results["generation_metrics"]
    for metric_name, value in gen_metrics.items():
        description = {
            "answer_relevancy": "addresses question",
        }.get(metric_name, "")

        # Handle list values (per-sample metrics)
        if isinstance(value, list):
            formatted_values = [f"{float(v):.2f}" for v in value]
            print(f"  {metric_name.capitalize()}: {formatted_values} ({description})")
        else:
            # Single aggregated value
            print(f"  {metric_name.capitalize()}: {value:.2f} "
                  f"({value*100:.0f}% {description})")
    print("=" * 70)


if __name__ == "__main__":
    main()
