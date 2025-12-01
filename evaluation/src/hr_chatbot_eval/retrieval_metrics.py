"""Retrieval quality metrics for HR Chatbot evaluation.

Compares retrieved source URLs against expected sources to measure
retrieval precision, recall, and F1 score.
"""

from urllib.parse import urlparse


def normalize_url(url: str) -> str:
    """
    Normalize URL for comparison.

    - Converts to lowercase
    - Strips trailing slashes
    - Removes query parameters and fragments
    - Keeps only scheme + netloc + path

    Args:
        url: URL string to normalize

    Returns:
        Normalized URL string
    """
    parsed = urlparse(url.lower().strip())
    # Reconstruct URL with scheme + netloc + path only
    normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    # Remove trailing slash
    return normalized.rstrip("/")


def calculate_source_metrics(
    retrieved_urls: list[str], expected_urls: list[str]
) -> dict:
    """
    Calculate precision, recall, and F1 for source URL retrieval.

    Args:
        retrieved_urls: List of URLs returned by the API
        expected_urls: List of URLs that should have been retrieved

    Returns:
        Dict with metrics:
        - precision: % of retrieved URLs that are correct
        - recall: % of expected URLs that were retrieved
        - f1: Harmonic mean of precision and recall
        - true_positives: Count of correct retrievals
        - false_positives: Count of incorrect retrievals
        - false_negatives: Count of missed expected URLs
    """
    # Normalize all URLs for comparison
    retrieved_set = {normalize_url(url) for url in retrieved_urls if url}
    expected_set = {normalize_url(url) for url in expected_urls if url}

    # Calculate set operations
    true_positives = len(retrieved_set & expected_set)
    false_positives = len(retrieved_set - expected_set)
    false_negatives = len(expected_set - retrieved_set)

    # Calculate metrics
    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )

    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def calculate_aggregate_metrics(per_sample_metrics: list[dict]) -> dict:
    """
    Calculate average metrics across multiple samples.

    Args:
        per_sample_metrics: List of metric dicts from calculate_source_metrics()

    Returns:
        Dict with averaged metrics:
        - avg_source_precision
        - avg_source_recall
        - total_samples
    """
    if not per_sample_metrics:
        return {
            "avg_source_precision": 0.0,
            "avg_source_recall": 0.0,
            "total_samples": 0,
        }

    total_precision = sum(m["precision"] for m in per_sample_metrics)
    total_recall = sum(m["recall"] for m in per_sample_metrics)
    num_samples = len(per_sample_metrics)

    return {
        "avg_source_precision": total_precision / num_samples,
        "avg_source_recall": total_recall / num_samples,
        "total_samples": num_samples,
    }
