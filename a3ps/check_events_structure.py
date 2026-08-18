import json
from pathlib import Path

# Sample events.json to understand structure
events_dir = Path('eval/anticipation')
if events_dir.exists():
    sample_files = list(events_dir.glob('*/events.json'))
    if sample_files:
        with open(sample_files[0]) as f:
            data = json.load(f)
        
        print(f"Sample file: {sample_files[0].parent.name}/events.json")
        print(f"Data type: {type(data).__name__}")
        
        if isinstance(data, dict):
            print(f"Top-level keys: {list(data.keys())}")
            if 'frames' in data:
                print(f"Number of frames: {len(data['frames'])}")
                if len(data['frames']) > 0:
                    frame = data['frames'][0]
                    print(f"First frame type: {type(frame).__name__}")
                    if isinstance(frame, dict):
                        print(f"First frame keys: {list(frame.keys())}")
        elif isinstance(data, list):
            print(f"List with {len(data)} items")
            if len(data) > 0:
                print(f"First item type: {type(data[0]).__name__}")
                if isinstance(data[0], dict):
                    print(f"First item keys: {list(data[0].keys())}")
        
        # Print first item/frame
        if isinstance(data, dict) and 'frames' in data and len(data['frames']) > 0:
            import json as json_module
            print("\nFirst frame content (truncated):")
            print(json_module.dumps(data['frames'][0], indent=2)[:500])
        elif isinstance(data, list) and len(data) > 0:
            import json as json_module
            print("\nFirst item content (truncated):")
            print(json_module.dumps(data[0], indent=2)[:500])
