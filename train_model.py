import ast
import os
import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "models")

MOVIES_FILE = os.path.join(DATASET_DIR, "movies.csv")
CREDITS_FILE = os.path.join(DATASET_DIR, "credits.csv")


def parse_names(value, limit=None):
    """Extract the name field from a TMDB JSON-like column."""
    try:
        items = ast.literal_eval(value) if isinstance(value, str) else value
        if not isinstance(items, list):
            return []
        names = [str(item.get("name", "")).strip() for item in items if isinstance(item, dict)]
        names = [name for name in names if name]
        return names[:limit] if limit else names
    except (ValueError, SyntaxError, TypeError):
        return []


def get_director(value):
    try:
        items = ast.literal_eval(value) if isinstance(value, str) else value
        if not isinstance(items, list):
            return ""
        for item in items:
            if isinstance(item, dict) and item.get("job") == "Director":
                return str(item.get("name", "")).strip()
    except (ValueError, SyntaxError, TypeError):
        pass
    return ""


def clean_token(value):
    return str(value).lower().replace(" ", "_").replace("-", "_")


def build_dataset():
    movies = pd.read_csv(MOVIES_FILE)
    credits = pd.read_csv(CREDITS_FILE)

    credits = credits.rename(columns={"movie_id": "id"})
    merged = movies.merge(credits[["id", "cast", "crew"]], on="id", how="left")

    merged["genres_list"] = merged["genres"].apply(parse_names)
    merged["keywords_list"] = merged["keywords"].apply(parse_names)
    merged["cast_list"] = merged["cast"].apply(lambda x: parse_names(x, 3))
    merged["director"] = merged["crew"].apply(get_director)

    def make_tags(row):
        overview = str(row.get("overview", "") or "")
        genres = [clean_token(x) for x in row["genres_list"]]
        keywords = [clean_token(x) for x in row["keywords_list"]]
        cast = [clean_token(x) for x in row["cast_list"]]
        director = clean_token(row["director"]) if row["director"] else ""
        return " ".join([overview] + genres * 2 + keywords + cast + ([director] if director else []))

    merged["tags"] = merged.apply(make_tags, axis=1)
    merged["title"] = merged["title"].fillna(merged["original_title"]).fillna("Unknown")
    merged["tags"] = merged["tags"].fillna("").str.strip()

    result = merged[
        [
            "id",
            "title",
            "tags",
            "genres_list",
            "overview",
            "release_date",
            "vote_average",
            "vote_count",
            "runtime",
            "original_language",
            "director",
        ]
    ].copy()
    result = result.rename(columns={"id": "movie_id", "genres_list": "genres"})
    result = result.drop_duplicates(subset=["movie_id"]).reset_index(drop=True)
    result = result[result["tags"].str.len() > 0].reset_index(drop=True)
    return result


def train():
    os.makedirs(MODEL_DIR, exist_ok=True)
    movies = build_dataset()

    vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
    tfidf_matrix = vectorizer.fit_transform(movies["tags"])

    # Because TF-IDF vectors are L2-normalized by default, their dot product
    # is cosine similarity. Keeping this sparse saves a large amount of space.
    similarity = linear_kernel(tfidf_matrix, tfidf_matrix, dense_output=False).tocsr().astype("float32")

    with open(os.path.join(MODEL_DIR, "movie_list.pkl"), "wb") as file:
        pickle.dump(movies, file, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(MODEL_DIR, "similarity.pkl"), "wb") as file:
        pickle.dump(similarity, file, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(MODEL_DIR, "vectorizer.pkl"), "wb") as file:
        pickle.dump(vectorizer, file, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Created {len(movies)} movies")
    print(f"Similarity matrix shape: {similarity.shape}")
    print(f"Models saved to: {MODEL_DIR}")


if __name__ == "__main__":
    train()
