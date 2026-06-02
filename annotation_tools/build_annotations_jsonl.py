import argparse
import json
import random
from pathlib import Path


def iter_layout_jsons(root):
    root = Path(root)
    for path in sorted(root.glob("sample_*/layout_annotations.json")):
        yield path


def write_jsonl(paths, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as out:
        for path in paths:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            out.write(json.dumps(data, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths = list(iter_layout_jsons(args.synthetic_root))

    random.seed(args.seed)
    random.shuffle(paths)

    n = len(paths)

    if n == 0:
        train_paths, val_paths, test_paths = [], [], []
    elif n < 3:
        train_paths = paths
        val_paths = []
        test_paths = []
    else:
        n_train = max(1, int(n * args.train_ratio))
        n_val = max(1, int(n * args.val_ratio))

        if n_train + n_val >= n:
            n_train = n - 2
            n_val = 1

        train_paths = paths[:n_train]
        val_paths = paths[n_train:n_train + n_val]
        test_paths = paths[n_train + n_val:]

    output_dir = Path(args.output_dir)
    write_jsonl(train_paths, output_dir / "annotations_train.jsonl")
    write_jsonl(val_paths, output_dir / "annotations_val.jsonl")
    write_jsonl(test_paths, output_dir / "annotations_test.jsonl")

    print(f"Total: {n}")
    print(f"Train: {len(train_paths)}")
    print(f"Val: {len(val_paths)}")
    print(f"Test: {len(test_paths)}")


if __name__ == "__main__":
    main()
