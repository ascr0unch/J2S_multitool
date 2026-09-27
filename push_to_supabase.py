"""
Script exécuté automatiquement par GitHub Actions.
Télécharge le dataset Kaggle bgg-data-full et upsert les jeux dans la
table Supabase "board_games".
"""

import glob
import math
import os
import time

import pandas as pd
import kagglehub
from supabase import create_client
from postgrest.exceptions import APIError

DATASET = "sylvainballerini/bgg-data-full"
BATCH_SIZE = 150
MAX_RETRIES = 4


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


def to_int(value):
    """Convertit en int propre : pandas lit souvent 10 comme 10.0 (float)
    dès qu'une colonne contient des cases vides ailleurs, or les colonnes
    Postgres "int" refusent les valeurs avec virgule."""
    if value is None:
        return None
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def to_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
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
            "year_published": to_int(pick(norm, ["yearpublished", "year"])),
            "min_players": to_int(pick(norm, ["minplayer", "minplayers"])),
            "max_players": to_int(pick(norm, ["maxplayer", "maxplayers"])),
            "min_age": to_int(pick(norm, ["minage"])),
            "play_time_minutes": to_int(pick(
                norm, ["playingtime", "avgplaytime", "maxplaytime", "minplaytime"]
            )),
            "rating": to_float(pick(norm, ["average", "ratingaverage", "avgrating"])),
            "complexity": to_float(
                pick(norm, ["averageweight", "complexity", "weight"])
            ),
            "rank": to_int(pick(norm, ["rank", "bggrank"])),
            "raw": record,
        })
    return rows


def push_to_supabase(rows):
    url = os.environ["SUPABASE_URL"]
    secret_key = os.environ["SUPABASE_SECRET_KEY"]
    supabase = create_client(url, secret_key)

    total = len(rows)
    for i in range(0, total, BATCH_SIZE):
        batch = rows[i:i + BATCH_SIZE]
        attempt = 0
        while True:
            attempt += 1
            try:
                supabase.table("board_games").upsert(
                    batch, on_conflict="external_id"
                ).execute()
                break
            except APIError as e:
                if attempt >= MAX_RETRIES:
                    print(
                        f"Échec définitif sur le lot {i}-{i + len(batch)} "
                        f"après {attempt} tentatives : {e}"
                    )
                    raise
                wait_seconds = attempt * 5
                print(
                    f"Erreur sur le lot {i}-{i + len(batch)} "
                    f"(tentative {attempt}/{MAX_RETRIES}) : {e}. "
                    f"Nouvelle tentative dans {wait_seconds}s..."
                )
                time.sleep(wait_seconds)
        print(f"Upsert {min(i + BATCH_SIZE, total)}/{total}")


def main():
    records = download_records()
    rows = build_rows(records)
    push_to_supabase(rows)
    print("Terminé : les jeux sont dans la table 'board_games' de Supabase.")


if __name__ == "__main__":
    main()
