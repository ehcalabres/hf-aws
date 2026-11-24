"""Sample inference payloads for SageMaker ModelBuilder."""

from typing import Any


def get_schema_samples(task: str, model_id: str) -> dict[str, Any]:
    """Sample payloads shared by ModelBuilder deployment and standalone code generation."""
    # Define sample inputs/outputs based on task
    task_schemas = {
        "text-generation": {
            "sample_input": {
                "model": model_id,
                "prompt": "Hi, what can you help me with?",
                "max_tokens": 256,
            },
            "sample_output": {"choices": [{"text": "Generated text."}]},
        },
        "text2text-generation": {
            "sample_input": {"inputs": "Translate to French: Hello, how are you?"},
            "sample_output": [{"generated_text": "Bonjour, comment allez-vous?"}],
        },
        "text-classification": {
            "sample_input": {"inputs": "I love this product!"},
            "sample_output": [{"label": "POSITIVE", "score": 0.99}],
        },
        "token-classification": {
            "sample_input": {"inputs": "My name is John and I live in New York."},
            "sample_output": [[{"entity": "PER", "word": "John", "score": 0.99}]],
        },
        "question-answering": {
            "sample_input": {
                "question": "What is the capital of France?",
                "context": "Paris is the capital of France.",
            },
            "sample_output": {"answer": "Paris", "score": 0.99},
        },
        "fill-mask": {
            "sample_input": {"inputs": "The capital of France is [MASK]."},
            "sample_output": [{"token_str": "Paris", "score": 0.99}],
        },
        "summarization": {
            "sample_input": {"inputs": "This is a long text that needs to be summarized."},
            "sample_output": [{"summary_text": "A short summary."}],
        },
        "translation": {
            "sample_input": {"inputs": "Hello, how are you?"},
            "sample_output": [{"translation_text": "Bonjour, comment allez-vous?"}],
        },
        "feature-extraction": {
            "sample_input": {"inputs": "Hello world"},
            "sample_output": [[0.1, 0.2, 0.3]],
        },
        "sentence-similarity": {
            "sample_input": {"inputs": {"source_sentence": "Hello", "sentences": ["Hi", "Goodbye"]}},
            "sample_output": [0.9, 0.1],
        },
    }

    return task_schemas.get(task, {"sample_input": {"inputs": "Hello"}, "sample_output": {}})
