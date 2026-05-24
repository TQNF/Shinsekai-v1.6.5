import os, sys
os.chdir(r'e:\Win_Work\AI-Shinsekai\Shinsekai-v1.6.5')
sys.path.insert(0, '.')

from qdrant_client import QdrantClient
from qdrant_client.models import PointVectors

CONV_PATH = "data/memory/qdrant_conv"
COLLECTION = "conversation_history"

client = QdrantClient(path=CONV_PATH)

try:
    count = client.count(COLLECTION).count
except Exception:
    print("Collection is empty or does not exist")
    sys.exit(0)

print(f"Total points: {count}")

all_pts, _ = client.scroll(collection_name=COLLECTION, limit=10000)
print(f"Scrolled: {len(all_pts)} points")

missing = 0
for i, pt in enumerate(all_pts):
    payload = pt.payload or {}
    if "chunk_index" not in payload:
        missing += 1
        client.set_payload(
            collection_name=COLLECTION,
            payload={"chunk_index": i},
            points=[str(pt.id)],
        )

print(f"Added chunk_index to {missing} points")
