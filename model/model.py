"""Train the baseline and positional context classifiers."""

from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from prepare_data import RESULTS_DIR, Occurrence, read_corpus, write_csv

SENTENCE_BOUNDARY = "<SAKINIO_RIBA>"


@dataclass(frozen=True)
class ContextWindow:
    key: str
    title: str
    left: int
    right: int


BASELINE = ContextWindow("baseline", "Visada jungtukas", 0, 0)
BASE_WINDOWS = (
    BASELINE,
    ContextWindow("both_1", "Po 1 iš abiejų pusių", 1, 1),
    ContextWindow("both_3", "Po 3 iš abiejų pusių", 3, 3),
)
Predictions = dict[str, list[str]]


def context_features(
    occurrence: Occurrence, window: ContextWindow
) -> dict[str, str]:
    features = {}
    left_nearest_first = occurrence.left[::-1]
    for side, length, words in (
        ("L", window.left, left_nearest_first),
        ("R", window.right, occurrence.right),
    ):
        for index in range(length):
            features[f"{side}{index + 1}"] = (
                words[index] if index < len(words) else SENTENCE_BOUNDARY
            )
    return features


def predict(
    train: list[Occurrence],
    held_out: list[Occurrence],
    windows: tuple[ContextWindow, ...],
) -> Predictions:
    predictions = {}
    train_labels = [occurrence.label for occurrence in train]
    for window in windows:
        if window == BASELINE:
            predictions[window.key] = ["CCONJ"] * len(held_out)
            continue
        classifier = make_pipeline(
            DictVectorizer(),
            LogisticRegression(solver="liblinear", random_state=42, max_iter=1000),
        )
        classifier.fit(
            [context_features(occurrence, window) for occurrence in train],
            train_labels,
        )
        held_out_features = [
            context_features(occurrence, window) for occurrence in held_out
        ]
        predictions[window.key] = classifier.predict(held_out_features).tolist()
    return predictions


def write_predictions(
    occurrences: list[Occurrence], predictions: Predictions, path: Path
) -> None:
    rows = (
        (occurrence.sentence_id, occurrence.token_id, occurrence.label, *labels)
        for occurrence, labels in zip(
            occurrences, zip(*predictions.values(), strict=True), strict=True
        )
    )
    write_csv(path, ("sentence_id", "token_id", "gold", *predictions), rows)


if __name__ == "__main__":
    parser = ArgumentParser(description="Generuoja bazinių variantų prognozes.")
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    corpus = read_corpus()
    held_out = corpus[args.split]
    path = args.output_dir / f"predictions_{args.split}.csv"
    write_predictions(held_out, predict(corpus["train"], held_out, BASE_WINDOWS), path)
    print(f"Prognozės: {path}")
