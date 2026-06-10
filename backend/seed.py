from __future__ import annotations

import asyncio
import json
import logging

import numpy as np
from sqlmodel import Session, select

from db import engine, init_db
from services.grid import encode_grid
from tables import MapTable, SpaceTable

_log = logging.getLogger(__name__)

WIDTH_CELLS = 100
HEIGHT_CELLS = 60
CELL_SIZE_M = 0.1


def _empty() -> np.ndarray:
    return np.zeros((HEIGHT_CELLS, WIDTH_CELLS), dtype=np.uint8)


def _grid_test_empty() -> np.ndarray:
    return _empty()


def _grid_test_lab() -> np.ndarray:
    grid = _empty()
    grid[30, 20:80] = 1
    return grid


def _grid_test_maze() -> np.ndarray:
    grid = _empty()
    grid[0:40, 25] = 1
    grid[20:60, 50] = 1
    grid[0:40, 75] = 1
    return grid


def _grid_test_apartment() -> np.ndarray:
    grid = _empty()

    grid[30, 0:20] = 1
    grid[30, 30:65] = 1
    grid[30, 75:100] = 1

    grid[0:23, 30] = 1
    grid[0:23, 65] = 1

    grid[30:50, 40] = 1
    grid[38:60, 75] = 1

    grid[3:8, 3:12] = 1
    grid[4:10, 35:43] = 1
    grid[4:12, 85:95] = 1
    grid[35:42, 22:32] = 1
    grid[50:55, 50:60] = 1
    grid[42:48, 85:92] = 1

    return grid


_DEMO_MAPS: list[dict] = [
    {
        "name": "test_apartment",
        "build": _grid_test_apartment,
        "spaces": [
            {
                "name": "база",
                "vertices": [(0.0, 0.0), (2.0, 0.0), (2.0, 1.5), (0.0, 1.5)],
                "is_home": True,
            },
            {
                "name": "офіс",
                "vertices": [(8.0, 4.0), (10.0, 4.0), (10.0, 6.0), (8.0, 6.0)],
            },
        ],
    },
    {
        "name": "test_empty",
        "build": _grid_test_empty,
        "spaces": [
            {
                "name": "база",
                "vertices": [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)],
                "is_home": True,
            },
            {
                "name": "вітальня",
                "vertices": [(2.0, 0.0), (5.0, 0.0), (5.0, 3.0), (2.0, 3.0)],
            },
            {
                "name": "кухня",
                "vertices": [(5.0, 0.0), (10.0, 0.0), (10.0, 3.0), (5.0, 3.0)],
            },
            {
                "name": "спальня",
                "vertices": [(0.0, 3.0), (10.0, 3.0), (10.0, 6.0), (0.0, 6.0)],
            },
        ],
    },
    {
        "name": "test_lab",
        "build": _grid_test_lab,
        "spaces": [
            {
                "name": "нижня кімната",
                "vertices": [(0.0, 0.0), (10.0, 0.0), (10.0, 3.0), (0.0, 3.0)],
                "is_home": True,
            },
            {
                "name": "верхня кімната",
                "vertices": [(0.0, 3.0), (10.0, 3.0), (10.0, 6.0), (0.0, 6.0)],
            },
        ],
    },
    {
        "name": "test_maze",
        "build": _grid_test_maze,
        "spaces": [
            {
                "name": "база-зона",
                "vertices": [(0.0, 0.0), (2.5, 0.0), (2.5, 6.0), (0.0, 6.0)],
                "is_home": True,
            },
            {
                "name": "ціль-зона",
                "vertices": [(7.5, 0.0), (10.0, 0.0), (10.0, 6.0), (7.5, 6.0)],
            },
        ],
    },
]


def seed(session: Session) -> int:
    if session.exec(select(MapTable).limit(1)).first() is not None:
        return 0

    for spec in _DEMO_MAPS:
        grid = spec["build"]()
        m = MapTable(
            name=spec["name"],
            width_cells=WIDTH_CELLS,
            height_cells=HEIGHT_CELLS,
            cell_size_m=CELL_SIZE_M,
            grid_data=encode_grid(grid),
        )
        session.add(m)
        session.flush()

        for space_spec in spec.get("spaces", []):
            verts = [(float(x), float(y)) for x, y in space_spec["vertices"]]
            session.add(
                SpaceTable(
                    map_id=m.id,
                    name=space_spec["name"],
                    vertices_json=json.dumps(verts),
                    is_home=bool(space_spec.get("is_home", False)),
                )
            )

    session.commit()
    _log.info("seeded %d demo maps", len(_DEMO_MAPS))
    return len(_DEMO_MAPS)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    asyncio.run(init_db())
    with Session(engine) as session:
        created = seed(session)
    if created:
        print(f"seeded {created} demo maps")
    else:
        print("DB already has maps; skipping seed")


if __name__ == "__main__":
    main()
