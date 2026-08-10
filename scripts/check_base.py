from datetime import UTC, datetime

from moirai.engine.data_fabric.ingestion.base import FetchResult
from moirai.engine.data_fabric.series.models import Source

result = FetchResult(
    source=Source.MANUAL,
    source_series_id="demo",
    url="file://demo",
    content=b'{"x": 1}',
    fetched_at=datetime.now(UTC),
)

print("hash:  ", result.content_hash[:32])
print("verify:", result.verify())
print("size:  ", result.size_bytes)
print("prov:  ", result.provenance()["content_hash"][:32])