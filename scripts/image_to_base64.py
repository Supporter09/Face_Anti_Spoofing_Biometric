import base64
import sys
from pathlib import Path

image_path = Path(sys.argv[1])
data = base64.b64encode(image_path.read_bytes()).decode("utf-8")
print(data)