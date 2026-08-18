import json
from pathlib import Path
from collections import defaultdict

# Section C: Confirm/refute CPU-side findings

print("=" * 80)
print("SECTION C: CONFIRM/REFUTE CPU-SIDE FINDINGS")
print("=" * 80)

# Find and organize events.json files
events_by_clip = {}

# Check eval/anticipation/
antici_dir = Path('eval/anticipation')
if antici_dir.exists():
    for clip_folder in antici_dir.iterdir():
        if clip_folder.is_dir():
            events_file = clip_folder / 'events.json'
            if events_file.exists():
                clip_id = clip_folder.name
                events_by_clip[clip_id] = str(events_file)

# Check dashboard/clips/
clips_dir = Path('dashboard/clips')
if clips_dir.exists():
    for events_file in clips_dir.rglob('events.json'):
        clip_id = events_file.parent.name
        if clip_id not in events_by_clip:
            events_by_clip[clip_id] = str(events_file)

print(f"\nFound {len(events_by_clip)} events.json files")

# 1. Probability saturation
print("\n## 1. Probability saturation")
collision_probs = []
collision_probs_1_0 = 0
total_events = 0

for clip_id, events_path in events_by_clip.items():
    try:
        with open(events_path, 'r') as f:
            data = json.load(f)
        
        # Check structure
        if isinstance(data, dict) and 'frames' in data:
            frames = data['frames']
        elif isinstance(data, list):
            frames = data
        else:
            continue
        
        for frame in frames:
            # Handle different frame structures
            if isinstance(frame, dict):
                # Check for events in frame
                events = frame.get('events', [])
                if not isinstance(events, list):
                    continue
                for event in events:
                    if isinstance(event, dict) and 'collision_prob' in event:
                        prob = event['collision_prob']
                        collision_probs.append(prob)
                        if prob == 1.0:
                            collision_probs_1_0 += 1
                        total_events += 1
    except Exception as e:
        pass

print(f"Total events with collision_prob: {total_events}")
if collision_probs:
    print(f"collision_prob distribution:")
    print(f"  min: {min(collision_probs):.4f}")
    print(f"  max: {max(collision_probs):.4f}")
    print(f"  mean: {sum(collision_probs)/len(collision_probs):.4f}")
    print(f"  fraction == 1.00: {collision_probs_1_0} / {total_events} = {collision_probs_1_0/total_events*100:.1f}%")
else:
    print("No collision_prob values found in events")

# 2. No anticipation - fires on first forecast step (ttc_s == 0.2)
print("\n## 2. No anticipation - fires on first forecast step")
ttc_values = []
ttc_0_2 = 0
total_ttc_events = 0

for clip_id, events_path in events_by_clip.items():
    try:
        with open(events_path, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, dict) and 'frames' in data:
            frames = data['frames']
        elif isinstance(data, list):
            frames = data
        else:
            continue
        
        for frame in frames:
            if isinstance(frame, dict):
                events = frame.get('events', [])
                if not isinstance(events, list):
                    continue
                for event in events:
                    if isinstance(event, dict) and 'ttc_s' in event:
                        ttc = event['ttc_s']
                        ttc_values.append(ttc)
                        if ttc == 0.2:
                            ttc_0_2 += 1
                        total_ttc_events += 1
    except Exception as e:
        pass

print(f"Total events with ttc_s: {total_ttc_events}")
if ttc_values:
    print(f"ttc_s distribution:")
    print(f"  min: {min(ttc_values):.4f}")
    print(f"  max: {max(ttc_values):.4f}")
    print(f"  mean: {sum(ttc_values)/len(ttc_values):.4f}")
    print(f"  fraction == 0.2: {ttc_0_2} / {total_ttc_events} = {ttc_0_2/total_ttc_events*100:.1f}%")
else:
    print("No ttc_s values found in events")

# 3. Check std_bev values
print("\n## 3. Uncertainty miscalibrated")
print("Looking for std_bev in events...")
std_bev_first = []
std_bev_last = []

for clip_id, events_path in events_by_clip.items():
    try:
        with open(events_path, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, dict) and 'frames' in data:
            frames = data['frames']
        elif isinstance(data, list):
            frames = data
        else:
            continue
        
        for frame in frames:
            if isinstance(frame, dict):
                events = frame.get('events', [])
                if not isinstance(events, list):
                    continue
                for event in events:
                    if isinstance(event, dict):
                        if 'prediction' in event and event['prediction'] is not None:
                            pred = event['prediction']
                            if isinstance(pred, dict):
                                if 'std_bev' in pred:
                                    std_bev = pred['std_bev']
                                    if isinstance(std_bev, list) and len(std_bev) > 0:
                                        std_bev_first.append(std_bev[0])
                                        if len(std_bev) > 1:
                                            std_bev_last.append(std_bev[-1])
    except Exception as e:
        pass

if std_bev_first:
    print(f"std_bev[0] (first step): n={len(std_bev_first)}")
    print(f"  mean: {sum(std_bev_first)/len(std_bev_first):.2f} px")
    print(f"  min: {min(std_bev_first):.2f}, max: {max(std_bev_first):.2f}")

if std_bev_last:
    print(f"std_bev[-1] (4s horizon): n={len(std_bev_last)}")
    print(f"  mean: {sum(std_bev_last)/len(std_bev_last):.2f} px")
    print(f"  min: {min(std_bev_last):.2f}, max: {max(std_bev_last):.2f}")
else:
    print("No std_bev values found")

print("\nFDE@4s from forecast_table.md: 122.55 px (Kalman-CV), 99.74 px (Seq2Seq-LSTM)")
if std_bev_last:
    ratio = 122.55 / (sum(std_bev_last)/len(std_bev_last))
    print(f"Ratio FDE@4s / mean(std_bev[-1]): {ratio:.1f}x")
