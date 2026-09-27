"""
Script exécuté automatiquement par GitHub Actions.
Télécharge le dataset Kaggle bgg-data-full et upsert les jeux dans la
table Supabase "board_games".
"""

import glob
import math
import os

import pandas as pd
import kagglehub
from supabase import create_client

DATASET = "sylvainballerini/bgg-data-full"
BATCH_SIZE = 500


def normalize(key) -> str:
    return "".join(ch for ch in str(key).lower() if ch.isalnum())


def clean_value(value):
    """Remplace NaN / Infinity (non valides en JSON) par None."""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def clean_record(record: dict) -> dict:
    return {k: clean_value(v) for k, v in record.items()}


def pick(row_norm: dict, keys: list):
    for k in keys:
        v = row_norm.get(k)
        if v is not None and str(v).strip().lower() not in ("", "nan", "none"):
            return v
    return None


def download_records():
    print(f"Téléchargement du dataset {DATASET} ...")
    dataset_path = kagglehub.dataset_download(DATASET)

    csv_files = glob.glob(os.path.join(dataset_path, "**", "*.csv"), recursive=True)
    if not csv_files:
        raise FileNotFoundError("Aucun CSV trouvé dans le dataset téléchargé.")
    csv_file = max(csv_files, key=os.path.getsize)
    print("Fichier CSV utilisé :", csv_file)

    df = pd.read_csv(csv_file)
    print("Colonnes disponibles :", list(df.columns))
    print(f"Nombre de jeux : {len(df)}")

    records = df.to_dict(orient="records")
    # Nettoyage fait sur les dicts Python (plus fiable que via pandas,
    # qui remet parfois NaN à la place de None pour les colonnes numériques).
    records = [clean_record(r) for r in records]
    return records


def build_rows(records):
    rows = []
    for record in records:
        norm = {normalize(k): v for k, v in record.items()}

        # Noms réels du dataset bgg-data-full : name_game, id_bgg,
        # min_player / max_player (singulier), etc. On garde aussi des
        # variantes plus courantes au cas où le dataset serait mis à jour
        # avec d'autres noms de colonnes.
        name = pick(norm, ["namegame", "name", "title"]) or "Sans nom"
        external_id = pick(norm, ["idbgg", "id", "bggid"]) or name

        rows.append({
            "external_id": str(external_id),
            "name": str(name),
            "year_published": pick(norm, ["yearpublished", "year"]),
            "min_players": pick(norm, ["minplayer", "minplayers"]),
            "max_players": pick(norm, ["maxplayer", "maxplayers"]),
            "min_age": pick(norm, ["minage"]),
            "play_time_minutes": pick(
                norm, ["playingtime", "avgplaytime", "maxplaytime", "minplaytime"]
            ),
            "rating": pick(norm, ["average", "ratingaverage", "avgrating"]),
            "complexity": pick(norm, ["averageweight", "complexity", "weight"]),
            "rank": pick(norm, ["rank", "bggrank"]),
            "raw": record,
        })
    return rows


def push_to_supabase(rows):
    url = os.environ["SUPABASE_URL"]
    secret_key = os.environ["SUPABASE_SECRET_KEY"]
    supabase = create_client(url, secret_key)

    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i:i + BATCH_SIZE]
        supabase.table("board_games").upsert(batch, on_conflict="external_id").execute()
        print(f"Upsert {min(i + BATCH_SIZE, len(rows))}/{len(rows)}")


def main():
    records = download_records()
    rows = build_rows(records)
    push_to_supabase(rows)
    print("Terminé : les jeux sont dans la table 'board_games' de Supabase.")


if __name__ == "__main__":
    main()
