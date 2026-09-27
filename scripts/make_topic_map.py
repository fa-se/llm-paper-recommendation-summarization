"""Projects the embeddings of all OpenAlex topics to 2D with UMAP for the demo's topic map (demo/static/topic_map.json).

One-off: the taxonomy doesn't change. Needs the db, and umap-learn, which is not a project dependency:
    uv run --env-file .env --with umap-learn scripts/make_topic_map.py
"""

import json
from pathlib import Path

import numpy as np
from sqlalchemy import select

from core.config import Settings
from core.database import create_session
from core.sqlalchemy_models import Field, Subfield, Topic

OUT = Path(__file__).parent.parent / "demo" / "static" / "topic_map.json"


def main():
    import umap  # optional dependency, see the docstring

    session = create_session(Settings.from_env())
    rows = session.execute(
        select(Topic.id, Topic.name, Subfield.name, Field.name, Topic.embedding)
        .join(Subfield, Topic.subfield_id == Subfield.id)
        .join(Field, Subfield.field_id == Field.id)
        .order_by(Topic.id)
    ).all()
    embeddings = np.array([row[4] for row in rows], dtype=np.float32)
    # cosine: the metric the pipeline uses to match topics; fixed seed, so that the map is reproducible
    xy = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42).fit_transform(embeddings)
    xy = (xy - xy.min(axis=0)) / (xy.max(axis=0) - xy.min(axis=0))  # to [0, 1]

    fields = sorted({row[3] for row in rows})
    subfields = sorted({row[2] for row in rows})
    topics = [
        [topic_id, round(float(x), 4), round(float(y), 4), name, subfields.index(subfield), fields.index(field)]
        for (topic_id, name, subfield, field, _), (x, y) in zip(rows, xy, strict=True)
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "columns": ["id", "x", "y", "name", "subfield", "field"],
                "topics": topics,
                "subfields": subfields,
                "fields": fields,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )
    print(f"Wrote {len(topics)} topics to {OUT} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
