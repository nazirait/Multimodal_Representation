# scripts/preprocess_visualgenome.py

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Any, List


def load_json(path: Path):
    print(f"Loading {path} ...")
    with path.open("r") as f:
        return json.load(f)


def build_caption_map(region_descriptions):
    """
    region_descriptions: list of dicts, each with:
      - image_id
      - regions: list of dicts with 'phrase' or 'region_description'
    returns: dict image_id -> list[str] (phrases)
    """
    imgid_to_phrases: Dict[int, List[str]] = {}

    for entry in region_descriptions:
        image_id = entry.get("image_id") or entry.get("id")
        if image_id is None:
            continue

        regions = entry.get("regions", [])
        phrases = []

        for r in regions:
            phrase = (
                r.get("phrase")
                or r.get("region_description")
                or r.get("description")
            )
            if phrase:
                phrase = phrase.strip()
                if phrase:
                    phrases.append(phrase)

        if phrases:
            imgid_to_phrases[image_id] = phrases

    print(f"Caption map built for {len(imgid_to_phrases)} images.")
    return imgid_to_phrases


def build_object_label_map(objects_data):
    """
    objects_data: list of dicts, each with:
      - image_id
      - objects: list of dicts with 'names': list[str]
    Returns:
      - imgid_to_labels: dict image_id -> sorted list of unique labels
      - label_freq: Counter of label frequencies
    """
    imgid_to_labels: Dict[int, List[str]] = {}
    label_freq = Counter()

    for entry in objects_data:
        image_id = entry.get("image_id") or entry.get("id")
        if image_id is None:
            continue

        objects = entry.get("objects", [])
        labels = set()

        for obj in objects:
            for name in obj.get("names", []):
                name = name.strip().lower()
                if name:
                    labels.add(name)
                    label_freq[name] += 1

        if labels:
            imgid_to_labels[image_id] = sorted(labels)

    print(f"Object label map built for {len(imgid_to_labels)} images.")
    print(f"Total unique object labels: {len(label_freq)}")
    return imgid_to_labels, label_freq


def build_predicate_map(relationships_data, top_k: int = 50):
    """
    relationships_data: list of dicts, each with:
      - image_id
      - relationships: list of dicts with 'predicate'
    Returns:
      - imgid_to_preds: dict image_id -> list of predicate strings
      - predicate_vocab: list of top-K predicates
      - pred_to_idx: mapping predicate -> index
    """
    imgid_to_preds: Dict[int, List[str]] = {}
    pred_counter = Counter()

    print("Building predicate map from relationships ...")

    for entry in relationships_data:
        image_id = entry.get("image_id") or entry.get("id")
        if image_id is None:
            continue

        rels = entry.get("relationships", [])
        preds = []
        for r in rels:
            p = r.get("predicate")
            if p:
                p = p.strip().lower()
                if p:
                    preds.append(p)
                    pred_counter[p] += 1

        if preds:
            imgid_to_preds[image_id] = preds

    print(f"Predicate map built for {len(imgid_to_preds)} images.")
    print(f"Total unique predicates: {len(pred_counter)}")

    # Top-K predicates as our 'graph modality' vocabulary
    predicate_vocab = [p for p, _ in pred_counter.most_common(top_k)]
    pred_to_idx = {p: i for i, p in enumerate(predicate_vocab)}

    print(f"Using top {top_k} predicates for graph features.")
    return imgid_to_preds, predicate_vocab, pred_to_idx, pred_counter


def find_image_path(images_root: Path, image_id: int):
    """
    Try to find the jpg file for a given image_id.
    Handles expected VG layout:
      images/VG_100K/<id>.jpg
      images/VG_100K_2/<id>.jpg
      images/<id>.jpg   (fallback)
    """
    fname = f"{image_id}.jpg"
    candidates = [
        images_root / "VG_100K" / fname,
        images_root / "VG_100K_2" / fname,
        images_root / fname,
    ]

    for p in candidates:
        if p.exists():
            return p

    return None


def main(args):
    # assume script is run from project root: python scripts/preprocess_visualgenome.py
    project_root = Path(__file__).resolve().parent.parent
    raw_dir = project_root / "data" / "raw" / "visualgenome"
    proc_dir = project_root / "data" / "processed" / "visualgenome"
    proc_dir.mkdir(parents=True, exist_ok=True)

    images_root = raw_dir / "images"

    # 1. Load raw JSONs
    image_data = load_json(raw_dir / "image_data.json")
    region_descriptions = load_json(raw_dir / "region_descriptions.json")
    objects_data = load_json(raw_dir / "objects.json")
    relationships_data = load_json(raw_dir / "relationships.json")

    # 2. Build maps
    imgid_to_phrases = build_caption_map(region_descriptions)
    imgid_to_labels, label_freq = build_object_label_map(objects_data)
    imgid_to_preds, predicate_vocab, pred_to_idx, pred_counter = build_predicate_map(
        relationships_data, top_k=args.graph_top_k
    )

    # 3. Build examples
    random.seed(args.seed)
    examples: List[Dict[str, Any]] = []

    for img in image_data:
        image_id = img.get("image_id") or img.get("id")
        if image_id is None:
            continue

        # we want text + labels; graph is optional but preferred
        if image_id not in imgid_to_phrases:
            continue
        if image_id not in imgid_to_labels:
            continue

        img_path = find_image_path(images_root, image_id)
        if img_path is None:
            continue

        phrases = imgid_to_phrases[image_id]
        labels = imgid_to_labels[image_id]

        if len(labels) < args.min_labels:
            continue

        # Build a single caption string from top-K region phrases
        max_phrases = args.max_phrases
        selected_phrases = phrases[:max_phrases]
        caption = ". ".join(selected_phrases)

        rel_path = img_path.relative_to(project_root).as_posix()

        # Graph features from predicates
        preds = imgid_to_preds.get(image_id, [])
        graph_vec = [0.0] * args.graph_top_k
        for p in preds:
            j = pred_to_idx.get(p)
            if j is not None:
                graph_vec[j] += 1.0

        # optional normalization by number of relationships
        total = sum(graph_vec)
        if total > 0.0:
            graph_vec = [v / total for v in graph_vec]

        example = {
            "image_id": image_id,
            "image_path": rel_path,      # relative to project root
            "caption": caption,
            "num_regions": len(phrases),
            "object_labels": labels,
            "graph_features": graph_vec,  # <-- NEW graph modality
        }
        examples.append(example)

        if args.max_images > 0 and len(examples) >= args.max_images:
            break

    print(f"Collected {len(examples)} examples that satisfy all constraints.")

    if not examples:
        print("No examples found. Check paths and filters.")
        return

    # 4. Shuffle + split
    random.shuffle(examples)
    n = len(examples)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    n_test = n - n_train - n_val

    train = examples[:n_train]
    val = examples[n_train:n_train + n_val]
    test = examples[n_train + n_val:]

    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    # Annotate split directly on each row
    for ex in train:
        ex["split"] = "train"
    for ex in val:
        ex["split"] = "val"
    for ex in test:
        ex["split"] = "test"

    all_examples = train + val + test

    # 5. Save as JSONL
    out_path = proc_dir / "visualgenome_dataset.jsonl"
    print(f"Writing {len(all_examples)} rows to {out_path} ...")
    with out_path.open("w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")

    # 6. Save stats (object labels + predicates)
    stats_path = proc_dir / "visualgenome_stats.json"
    top_labels = label_freq.most_common(50)
    top_predicates = pred_counter.most_common(args.graph_top_k)

    stats = {
        "num_examples": len(all_examples),
        "num_unique_labels": len(label_freq),
        "top_50_labels": top_labels,
        "graph_top_k": args.graph_top_k,
        "predicate_vocab": [p for p, _ in top_predicates],
    }
    with stats_path.open("w") as f:
        json.dump(stats, f, indent=2)

    print(f"Saved stats to {stats_path}")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess Visual Genome into a subset JSONL file with image, text, labels, and graph features."
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=10000,
        help="Maximum number of images to keep (0 = all).",
    )
    parser.add_argument(
        "--min-labels",
        type=int,
        default=1,
        help="Minimum number of object labels per image.",
    )
    parser.add_argument(
        "--max-phrases",
        type=int,
        default=10,
        help="Maximum number of region phrases to join into caption.",
    )
    parser.add_argument(
        "--graph-top-k",
        type=int,
        default=50,
        help="Number of most frequent predicates to use as graph feature dimensions.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling.",
    )

    args = parser.parse_args()
    main(args)
