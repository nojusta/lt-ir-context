import csv
from argparse import ArgumentParser
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
SPLITS = ("train", "dev", "test")
LABELS = ("CCONJ", "PART")
LABEL_NAMES = {"CCONJ": "Jungtukas", "PART": "Dalelytė"}
MAX_CONTEXT = 3


@dataclass(frozen=True)
class Occurrence:
    sentence_id: str
    token_id: int
    sentence: str
    label: str
    left: tuple[str, ...]
    right: tuple[str, ...]


def read_split(path: Path) -> list[Occurrence]:
    occurrences = []
    for block in path.read_text(encoding="utf-8").split("\n\n"):
        lines = block.splitlines()
        metadata = dict(
            line[2:].split(" = ", 1)
            for line in lines
            if line.startswith("# ") and " = " in line
        )
        tokens = []
        for line in lines:
            columns = line.split("\t")
            # Integer IDs identify real tokens. The other IDs encode ranges
            # or empty nodes. Punctuation and symbols do not count as words.
            if columns[0].isdigit() and columns[3] not in {"PUNCT", "SYM"}:
                tokens.append((int(columns[0]), columns[1].casefold(), columns[3]))

        forms = [form for _, form, _ in tokens]
        for index, (token_id, form, label) in enumerate(tokens):
            if form == "ir" and label in LABELS:
                occurrences.append(
                    Occurrence(
                        sentence_id=metadata["sent_id"],
                        token_id=token_id,
                        sentence=metadata["text"],
                        label=label,
                        left=tuple(forms[max(0, index - MAX_CONTEXT) : index]),
                        right=tuple(forms[index + 1 : index + 1 + MAX_CONTEXT]),
                    )
                )
    return occurrences


def read_corpus() -> dict[str, list[Occurrence]]:
    return {
        split: read_split(DATA_DIR / f"lt_alksnis-ud-{split}.conllu")
        for split in SPLITS
    }


def write_csv(
    path: Path, header: Iterable[str], rows: Iterable[Iterable[object]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def write_counts(corpus: dict[str, list[Occurrence]], output_dir: Path) -> None:
    rows = []
    for split, occurrences in corpus.items():
        counts = Counter(occurrence.label for occurrence in occurrences)
        rows.append((split, counts["CCONJ"], counts["PART"], len(occurrences)))
    write_csv(output_dir / "data_counts.csv", ("split", *LABELS, "total"), rows)


if __name__ == "__main__":
    parser = ArgumentParser(description="Suskaičiuoja „ir“ atvejus ALKSNIS dalyse.")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    write_counts(read_corpus(), args.output_dir)
    print(f"Duomenų kiekiai: {args.output_dir / 'data_counts.csv'}")
