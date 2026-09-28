"""
Script exécuté automatiquement par GitHub Actions.
Télécharge le dataset Kaggle bgg-data-full (jeux + catégories + mécaniques +
éditeurs + familles) et upsert le tout dans la table Supabase "board_games".
"""

import glob
import math
import os
import time
import urllib.parse

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


def to_id_str(value):
    """Normalise un identifiant numérique (10 ou 10.0) en '10'."""
    if value is None:
        return None
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return None


def find_file(dataset_path, filename):
    matches = glob.glob(os.path.join(dataset_path, "**", filename), recursive=True)
    return matches[0] if matches else None


def load_lookup(path, id_candidates, name_candidates):
    """Charge un fichier de référence (ex: category.csv) en dict {id: nom}."""
    if not path:
        return {}
    df = pd.read_csv(path)
    norm_cols = {normalize(c): c for c in df.columns}
    id_col = next((norm_cols[c] for c in id_candidates if c in norm_cols), None)
    name_col = next((norm_cols[c] for c in name_candidates if c in norm_cols), None)
    if not id_col or not name_col:
        print(f"  ! Colonnes non trouvées dans {path} (colonnes : {list(df.columns)})")
        return {}
    df = df[[id_col, name_col]].dropna()
    ids = df[id_col].apply(to_id_str)
    return dict(zip(ids, df[name_col].astype(str)))


def load_junction(path, game_id_candidates, ref_id_candidates):
    """Charge un fichier de jointure (ex: b_game_category.csv) en dict
    {game_id: [ref_id, ref_id, ...]}."""
    if not path:
        return {}
    df = pd.read_csv(path)
    norm_cols = {normalize(c): c for c in df.columns}
    game_col = next((norm_cols[c] for c in game_id_candidates if c in norm_cols), None)
    ref_col = next((norm_cols[c] for c in ref_id_candidates if c in norm_cols), None)
    if not game_col or not ref_col:
        print(f"  ! Colonnes non trouvées dans {path} (colonnes : {list(df.columns)})")
        return {}
    df = df[[game_col, ref_col]].dropna()
    df["_gid"] = df[game_col].apply(to_id_str)
    df["_rid"] = df[ref_col].apply(to_id_str)
    df = df.dropna(subset=["_gid", "_rid"])
    return df.groupby("_gid")["_rid"].apply(list).to_dict()


def load_related_data(dataset_path):
    print("Chargement des catégories / mécaniques / éditeurs / familles ...")

    categories = load_lookup(
        find_file(dataset_path, "category.csv"),
        ["idcategory", "id"], ["category", "namecategory", "name"],
    )
    mechanics = load_lookup(
        find_file(dataset_path, "mechanic.csv"),
        ["idmechanic", "id"], ["mechanic", "namemechanic", "name"],
    )
    publishers = load_lookup(
        find_file(dataset_path, "publisher.csv"),
        ["idpublisher", "id"], ["publisher", "namepublisher", "name"],
    )
    families = load_lookup(
        find_file(dataset_path, "family.csv"),
        ["idfamily", "id"], ["family", "namefamily", "name"],
    )

    game_categories = load_junction(
        find_file(dataset_path, "b_game_category.csv"),
        ["idbgg", "idgame", "id"], ["idcategory"],
    )
    game_mechanics = load_junction(
        find_file(dataset_path, "b_game_mechanic.csv"),
        ["idbgg", "idgame", "id"], ["idmechanic"],
    )
    game_publishers = load_junction(
        find_file(dataset_path, "b_game_publisher.csv"),
        ["idbgg", "idgame", "id"], ["idpublisher"],
    )
    game_families = load_junction(
        find_file(dataset_path, "b_game_family.csv"),
        ["idbgg", "idgame", "id"], ["idfamily"],
    )

    print(
        f"  {len(categories)} catégories, {len(mechanics)} mécaniques, "
        f"{len(publishers)} éditeurs, {len(families)} familles."
    )

    return {
        "categories": categories,
        "mechanics": mechanics,
        "publishers": publishers,
        "families": families,
        "game_categories": game_categories,
        "game_mechanics": game_mechanics,
        "game_publishers": game_publishers,
        "game_families": game_families,
    }


def download_dataset():
    print(f"Téléchargement du dataset {DATASET} ...")
    return kagglehub.dataset_download(DATASET)


def load_game_records(dataset_path):
    csv_file = find_file(dataset_path, "game.csv")
    if not csv_file:
        # au cas où le fichier principal n'aurait pas exactement ce nom
        csv_files = glob.glob(os.path.join(dataset_path, "**", "*.csv"), recursive=True)
        csv_file = max(csv_files, key=os.path.getsize)
    print("Fichier jeux utilisé :", csv_file)

    df = pd.read_csv(csv_file)
    print("Colonnes disponibles :", list(df.columns))
    print(f"Nombre de jeux : {len(df)}")

    records = df.to_dict(orient="records")
    records = [clean_record(r) for r in records]
    return records


def build_rows(records, related):
    rows = []
    for record in records:
        norm = {normalize(k): v for k, v in record.items()}

        name = pick(norm, ["namegame", "name", "title"]) or "Sans nom"
        bgg_id_raw = pick(norm, ["idbgg", "id", "bggid"])
        bgg_id_str = to_id_str(bgg_id_raw)
        external_id = bgg_id_str or name

        rules_url = (
            f"https://boardgamegeek.com/boardgame/{bgg_id_str}/files"
            if bgg_id_str else None
        )
        search_query = urllib.parse.quote(
            f'"{name}" règles du jeu filetype:pdf français'
        )
        rules_search_url = f"https://www.google.com/search?q={search_query}"

        def resolve(junction_key, lookup_key):
            if not bgg_id_str:
                return None
            ref_ids = related[junction_key].get(bgg_id_str, [])
            lookup = related[lookup_key]
            names = [lookup[r] for r in ref_ids if r in lookup]
            return names or None

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
            "rules_url": rules_url,
            "rules_search_url": rules_search_url,
            "categories": resolve("game_categories", "categories"),
            "mechanics": resolve("game_mechanics", "mechanics"),
            "publishers": resolve("game_publishers", "publishers"),
            "families": resolve("game_families", "families"),
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
    dataset_path = download_dataset()
    records = load_game_records(dataset_path)
    related = load_related_data(dataset_path)
    rows = build_rows(records, related)
    push_to_supabase(rows)
    print("Terminé : les jeux (+ catégories/mécaniques/éditeurs/familles) "
          "sont dans la table 'board_games' de Supabase.")


if __name__ == "__main__":
    main()
