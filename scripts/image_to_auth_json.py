import base64
import json
import sys
from pathlib import Path


if len(sys.argv) < 4:
    print("Usage: python scripts/image_to_auth_json.py <image_path> <user_id> <output_json_path>")
    sys.exit(1)


image_path = Path(sys.argv[1])
user_id = sys.argv[2]
output_json_path = Path(sys.argv[3])

if not image_path.exists():
    print(f"Image not found: {image_path}")
    sys.exit(1)

image_base64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")

payload = {
    "user_id": user_id,
    "image_base64": image_base64,
}

output_json_path.parent.mkdir(parents=True, exist_ok=True)
output_json_path.write_text(
    json.dumps(payload, indent=2),
    encoding="utf-8",
)

print(f"Saved request JSON to: {output_json_path}")