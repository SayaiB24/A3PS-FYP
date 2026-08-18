import csv
import json
import statistics
from pathlib import Path
from collections import Counter

# Section B: Accuracy metrics

print("=" * 80)
print("SECTION B: ACCURACY METRICS")
print("=" * 80)

# From anticipation.md (already parsed):
print("\n## Headline numbers from anticipation.md")
print("A3PS (proactive):")
print("  - detection rate: 0.917 (55/60)")
print("  - false-alarm rate: 0.767 (46/60)")
print("  - mTTA: 16.11 s (n=55)")
print()
print("Reactive-proximity (baseline):")
print("  - detection rate: 0.967 (58/60)")
print("  - false-alarm rate: 0.783 (47/60)")
print("  - mTTA: 17.92 s (n=58)")
print()
print("Matched-subset mTTA (54 clips):")
print("  - A3PS: 16.17 s")
print("  - Reactive: 18.24 s")
print("  - Anticipation gain: A3PS is 2.07 s LATER (n=54)")
print()
print("A3PS AP (peak-prob ranking): 0.978")

# Read CSV for distribution analysis
print("\n## Distribution of a3ps_tta_s for anticipated positives")
with open('eval/anticipation_per_clip.csv', 'r') as f:
    reader = csv.DictReader(f)
    tta_values = []
    tta_negative_count = 0
    
    for row in reader:
        if row['a3ps_outcome'] == 'anticipated':
            try:
                tta = float(row['a3ps_tta_s'])
                tta_values.append(tta)
                if tta < 1.0:
                    tta_negative_count += 1
            except (ValueError, TypeError):
                pass

tta_values.sort()
print(f"Total anticipated positives with valid TTA: {len(tta_values)}")
if len(tta_values) > 0:
    print(f"  min: {min(tta_values):.3f}")
    print(f"  p25: {statistics.quantiles(tta_values, n=4)[0]:.3f}")
    print(f"  median: {statistics.median(tta_values):.3f}")
    print(f"  p75: {statistics.quantiles(tta_values, n=4)[2]:.3f}")
    print(f"  max: {max(tta_values):.3f}")
    print(f"  mean: {statistics.mean(tta_values):.3f}")

print(f"\nAnticipated positives with a3ps_tta_s < 1.0 (fired at impact): {tta_negative_count}")

# Now check for negatives with tracked actors
# Need to check events.json files in eval/anticipation/ and dashboard/clips/
print("\n## False-alarm rate restricted to negatives with tracked actors")
print("Checking events.json for tracked actor counts...")

events_paths = []
# Check eval/anticipation/
antici_dir = Path('eval/anticipation')
if antici_dir.exists():
    for json_file in antici_dir.glob('*.json'):
        events_paths.append(json_file)

# Check dashboard/clips/
clips_dir = Path('dashboard/clips')
if clips_dir.exists():
    for json_file in clips_dir.rglob('events.json'):
        events_paths.append(json_file)

print(f"Found {len(events_paths)} events.json files")

# Read CSV again and match with events
false_alarm_negatives = {}
with open('eval/anticipation_per_clip.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['label'] == 'neg':
            clip_id = row['clip_id']
            false_alarm_negatives[clip_id] = row['a3ps_outcome']

print(f"Total negative clips: {len(false_alarm_negatives)}")
print(f"Negative clip outcomes: {Counter(false_alarm_negatives.values())}")
