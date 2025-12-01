"""Dataset validation for HR Chatbot evaluation.

Validates evaluation dataset structure and quality before running evaluation.
"""

import logging
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def validate_dataset(samples: list[dict]) -> list[str]:
    """
    Validate evaluation dataset structure and content.

    Args:
        samples: List of sample dicts from eval_dataset.json

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []

    if not samples:
        errors.append("Dataset is empty - no samples to evaluate")
        return errors

    # Track questions to detect duplicates
    seen_questions = set()

    for idx, sample in enumerate(samples):
        sample_ref = f"Sample {idx + 1}"

        # Required fields
        if "question" not in sample:
            errors.append(f"{sample_ref}: Missing required field 'question'")
            continue

        question = sample["question"]

        # Check for empty question
        if not question or not question.strip():
            errors.append(f"{sample_ref}: Question is empty")

        # Check for duplicate questions
        if question in seen_questions:
            errors.append(f"{sample_ref}: Duplicate question '{question[:50]}...'")
        seen_questions.add(question)

        # Validate expected_sources (should be list of valid URLs)
        if "expected_sources" in sample:
            expected_sources = sample["expected_sources"]
            if not isinstance(expected_sources, list):
                errors.append(f"{sample_ref}: 'expected_sources' must be a list")
            elif not expected_sources:
                logger.warning(
                    f"{sample_ref}: 'expected_sources' is empty - retrieval metrics will be zero"
                )
            else:
                for url_idx, url in enumerate(expected_sources):
                    if not isinstance(url, str):
                        errors.append(
                            f"{sample_ref}: Expected source {url_idx + 1} is not a string"
                        )
                    elif not _is_valid_url(url):
                        errors.append(
                            f"{sample_ref}: Expected source {url_idx + 1} "
                            f"is not a valid URL: '{url}'"
                        )
        else:
            logger.warning(
                f"{sample_ref}: Missing 'expected_sources' - retrieval metrics will be unavailable"
            )

        # Validate ground_truth (optional but recommended)
        if "ground_truth" not in sample or not sample["ground_truth"]:
            logger.warning(
                f"{sample_ref}: Missing 'ground_truth' - context_recall metric will be unavailable"
            )

    return errors


def _is_valid_url(url: str) -> bool:
    """
    Check if a string is a valid URL.

    Args:
        url: URL string to validate

    Returns:
        True if valid URL, False otherwise
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False
