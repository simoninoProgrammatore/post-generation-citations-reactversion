"""Router per gestire i dataset ALCE (lista file disponibili, carica esempi)."""

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

from models.schemas import DatasetInfo, DatasetListResponse, DatasetLoadResponse

router = APIRouter(prefix="/api/dataset", tags=["dataset"])

# Path della cartella dati (relativo alla root del backend)
DATA_DIR = Path(__file__).parent.parent / "data" / "alce"


@router.get("/list", response_model=DatasetListResponse)
async def list_datasets():
    """Lista tutti i file .json dentro data/alce/."""
    if not DATA_DIR.exists():
        return DatasetListResponse(datasets=[])

    datasets = []
    for f in sorted(DATA_DIR.glob("*.json")):
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            n = len(data) if isinstance(data, list) else 0
            datasets.append(DatasetInfo(filename=f.name, num_examples=n))
        except Exception:
            datasets.append(DatasetInfo(filename=f.name, num_examples=-1))

    return DatasetListResponse(datasets=datasets)


@router.get("/load/{filename}", response_model=DatasetLoadResponse)
async def load_dataset(filename: str):
    """Carica un file dataset specifico. La filename deve essere in data/alce/."""
    # Validation: no path traversal
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Filename non valido")

    path = DATA_DIR / filename
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail=f"Dataset {filename} non trovato")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise HTTPException(status_code=422, detail="Il dataset deve essere una lista")
        return DatasetLoadResponse(filename=filename, examples=data)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=422, detail=f"JSON invalido: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))