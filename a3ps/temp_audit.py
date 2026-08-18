from pathlib import Path
import json

clips_dir = Path('dashboard/clips')

# Count folders
folders = [d for d in clips_dir.iterdir() if d.is_dir()]
print(f'Total folders: {len(folders)}')

# Count numeric vs dev*
numeric = 0
dev_prefix = 0
for f in folders:
    if f.name.isdigit():
        numeric += 1
    elif f.name.startswith('dev'):
        dev_prefix += 1

print(f'Numeric IDs: {numeric}')
print(f'dev* folders: {dev_prefix}')

# Count folders with annotated.mp4
with_video = 0
for f in folders:
    if (f / 'annotated.mp4').exists():
        with_video += 1

print(f'Folders with annotated.mp4: {with_video}')

# Show manifest.json contents
manifest_file = clips_dir / 'manifest.json'
if manifest_file.exists():
    with open(manifest_file) as f:
        manifest = json.load(f)
    print()
    print('manifest.json type:', type(manifest).__name__)
    if isinstance(manifest, list):
        print(f'  items: {len(manifest)}')
        if len(manifest) > 0:
            print(f'  first item keys: {list(manifest[0].keys())}')
    elif isinstance(manifest, dict):
        print('  keys:', list(manifest.keys()))
else:
    print('manifest.json: NOT FOUND')
