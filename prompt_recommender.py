"""Small adapter for the responsible-prompting recommendation classifier."""

import json
import re
from functools import lru_cache
from urllib.request import urlopen

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
PROMPT_VALUES_URL = (
    "https://raw.githubusercontent.com/IBM/responsible-prompting-api/"
    "main/prompt-sentences-main/prompt_sentences-all-minilm-l6-v2.json"
)


@lru_cache(maxsize=1)
def _resources():
    model = SentenceTransformer(MODEL_ID)
    with urlopen(PROMPT_VALUES_URL) as response:
        values = json.load(response)
    return model, values


def _similarity(left, right):
    left = np.asarray(left)
    right = np.asarray(right)
    return float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)))


def _sentences(text):
    return re.split(r"(?<=[.!?]) +", text)


def recommend_prompt(
    prompt,
    add_lower_threshold=0.1,
    add_upper_threshold=1.0,
    remove_lower_threshold=0.0,
    remove_upper_threshold=1.0,
    model_id=MODEL_ID,
):
    """Return the notebook-compatible add/remove recommendations."""
    if model_id != MODEL_ID:
        raise ValueError(f"Only {MODEL_ID} is supported")

    model, prompt_values = _resources()
    sentences = _sentences(prompt)
    embeddings = model.encode(sentences)
    output = {"add": [], "remove": []}

    last_embedding = embeddings[-1]
    for value in prompt_values["positive_values"]:
        if _similarity(last_embedding, value["centroid"]) > add_lower_threshold:
            candidates = [
                (p, _similarity(last_embedding, p["embedding"]))
                for p in value["prompts"]
            ]
            candidates = [
                (p, similarity)
                for p, similarity in candidates
                if add_lower_threshold < similarity < add_upper_threshold
            ]
            if candidates:
                prompt_value, similarity = max(candidates, key=lambda item: item[1])
                output["add"].append({"value": value["label"], "similarity": similarity})

    for sentence, embedding in zip(sentences, embeddings):
        for value in prompt_values["negative_values"]:
            if _similarity(embedding, value["centroid"]) > remove_lower_threshold:
                candidates = [
                    (p, _similarity(embedding, p["embedding"]))
                    for p in value["prompts"]
                ]
                candidates = [
                    (p, similarity)
                    for p, similarity in candidates
                    if similarity > remove_upper_threshold
                ]
                if candidates:
                    prompt_value, similarity = max(candidates, key=lambda item: item[1])
                    output["remove"].append({
                        "value": value["label"],
                        "similarity": similarity,
                        "sentence": sentence,
                    })

    for key in ("add", "remove"):
        output[key].sort(key=lambda item: item["similarity"], reverse=True)
        unique = []
        seen = set()
        for item in output[key]:
            if item["value"] not in seen:
                seen.add(item["value"])
                unique.append(item)
        output[key] = unique[:5]
    return output


def recommended_values(text):
    """Return the value associated with the highest similarity recommendation."""
    recommendations = recommend_prompt(text)
    items = recommendations.get("add", []) + recommendations.get("remove", [])
    best = max(items, key=lambda item: item.get("similarity", float("-inf")), default=None)
    return best.get("value", "") if best else ""