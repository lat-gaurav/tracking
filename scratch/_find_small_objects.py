import csv
from pathlib import Path

MOT_ROOT = Path("resources/visdrone_mot")
splits = {
    "train":    "VisDrone2019-MOT-train",
    "val":      "VisDrone2019-MOT-val",
    "test-dev": "VisDrone2019-MOT-test-dev",
}

results = []
for split_name, split_dir in splits.items():
    ann_dir = MOT_ROOT / split_dir / "annotations"
    if not ann_dir.exists():
        continue
    for ann_file in sorted(ann_dir.glob("*.txt")):
        areas = []
        with ann_file.open() as f:
            for row in csv.reader(f):
                if len(row) < 8:
                    continue
                cat = int(row[7])
                if cat == 0:
                    continue
                w, h = int(row[4]), int(row[5])
                if w > 0 and h > 0:
                    areas.append(w * h)
        if not areas:
            continue
        avg_area    = sum(areas) / len(areas)
        median_area = sorted(areas)[len(areas) // 2]
        pct_tiny    = sum(1 for a in areas if a < 400) / len(areas) * 100  # < 20x20 px
        results.append((avg_area, median_area, pct_tiny, len(areas), split_name, ann_file.stem))

results.sort(key=lambda x: x[0])
print(f"{'Rank':<5} {'Split':<10} {'Sequence':<32} {'AvgArea':>9} {'Median':>8} {'%Tiny<20x20':>12} {'NBoxes':>7}")
print("-" * 85)
for i, (avg, med, pct, n, sp, seq) in enumerate(results[:20], 1):
    print(f"{i:<5} {sp:<10} {seq:<32} {avg:>9.0f} {med:>8.0f} {pct:>11.1f}% {n:>7}")

print()
best = results[0]
print(f"BEST: {best[4]}/{best[5]}  avg_area={best[0]:.0f}px²  {best[2]:.1f}% boxes < 20×20")
