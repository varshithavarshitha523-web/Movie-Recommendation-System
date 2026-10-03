import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
import difflib

from concurrent.futures import ThreadPoolExecutor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Netflix Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>

/* MAIN */
.stApp {
    background-color: #141414;
    color: white;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background-color: #000000;
}

section[data-testid="stSidebar"] * {
    color: white;
}

/* LOGO */
.logo {
    color: #E50914;
    font-size: 32px;
    font-weight: 900;
    margin-bottom: 25px;
}

/* TITLES */
.main-title {
    color: #E50914;
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 5px;
}

.subtitle {
    color: #b3b3b3;
    font-size: 17px;
    margin-bottom: 25px;
}

.section-title {
    color: white;
    font-size: 28px;
    font-weight: 700;
    margin-top: 28px;
    margin-bottom: 18px;
}

/* MOVIE CARD */
.movie-card {
    background-color: #1f1f1f;
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 20px;
    transition: 0.2s;
}

.movie-card:hover {
    transform: scale(1.02);
}

/* MOVIE TITLE */
.movie-title {
    color: white;
    font-size: 16px;
    font-weight: 600;
    min-height: 42px;
    margin-top: 8px;
}

/* BUTTON */
.stButton > button {
    width: 100%;
    border-radius: 6px;
    border: none;
    background-color: #E50914;
    color: white;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #b20710;
    color: white;
}

/* SEARCH */
div[data-baseweb="input"] {
    background-color: #222222;
}

/* DIVIDER */
hr {
    border-color: #333333;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MOVIES
# ============================================================

@st.cache_resource
def load_movies():

    with open(
        "models/movie_list.pkl",
        "rb"
    ) as file:

        data = pickle.load(file)

    data = data.reset_index(drop=True)

    data["title"] = (
        data["title"]
        .fillna("")
        .astype(str)
    )

    data["tags"] = (
        data["tags"]
        .fillna("")
        .astype(str)
    )

    return data


movies = load_movies()


# ============================================================
# BUILD FAST TF-IDF MODEL
# ============================================================

@st.cache_resource
def build_tfidf_model():

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=10000,
        dtype=np.float32
    )

    vectors = vectorizer.fit_transform(
        movies["tags"]
    )

    return vectorizer, vectors


vectorizer, movie_vectors = build_tfidf_model()


# ============================================================
# OMDB CONFIG
# ============================================================

try:

    OMDB_API_KEY = st.secrets["OMDB_API_KEY"]

except Exception:

    OMDB_API_KEY = ""


OMDB_URL = "https://www.omdbapi.com/"


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:

    st.session_state.page = "Home"


if "selected_movie" not in st.session_state:

    st.session_state.selected_movie = None


if "recommendation_source" not in st.session_state:

    st.session_state.recommendation_source = None


if "favorites" not in st.session_state:

    st.session_state.favorites = []


# ============================================================
# OPEN MOVIE
# ============================================================

def open_movie(title):

    st.session_state.selected_movie = title

    st.session_state.recommendation_source = title

    st.session_state.page = "Details"

    st.rerun()


# ============================================================
# FAVORITES
# ============================================================

def add_favorite(title):

    if title not in st.session_state.favorites:

        st.session_state.favorites.append(title)


def remove_favorite(title):

    if title in st.session_state.favorites:

        st.session_state.favorites.remove(title)


def is_favorite(title):

    return title in st.session_state.favorites


# ============================================================
# OMDB REQUEST
# ============================================================

@st.cache_data(
    ttl=86400,
    max_entries=2000,
    show_spinner=False
)
def omdb_request(title):

    if not OMDB_API_KEY:

        return None

    try:

        response = requests.get(
            OMDB_URL,
            params={
                "apikey": OMDB_API_KEY,
                "t": title,
                "type": "movie",
                "plot": "full"
            },
            timeout=4
        )

        if response.status_code != 200:

            return None

        data = response.json()

        if data.get("Response") == "True":

            return data

    except Exception:

        pass

    return None


# ============================================================
# OMDB SEARCH
# ============================================================

@st.cache_data(
    ttl=86400,
    max_entries=2000,
    show_spinner=False
)
def omdb_search(title):

    if not OMDB_API_KEY:

        return None

    try:

        response = requests.get(
            OMDB_URL,
            params={
                "apikey": OMDB_API_KEY,
                "s": title,
                "type": "movie"
            },
            timeout=4
        )

        if response.status_code != 200:

            return None

        data = response.json()

        if data.get("Response") == "True":

            return data

    except Exception:

        pass

    return None


# ============================================================
# MOVIE DETAILS
# ============================================================

def get_movie_details(title):

    # Exact search
    data = omdb_request(title)

    if data:

        return data


    # Fallback search
    search = omdb_search(title)

    if not search:

        return None


    results = search.get(
        "Search",
        []
    )


    if not results:

        return None


    best = None
    best_score = 0


    for result in results:

        result_title = result.get(
            "Title",
            ""
        )

        score = difflib.SequenceMatcher(
            None,
            title.lower(),
            result_title.lower()
        ).ratio()


        if score > best_score:

            best_score = score

            best = result


    if not best:

        return None


    imdb_id = best.get(
        "imdbID"
    )


    if not imdb_id:

        return None


    try:

        response = requests.get(
            OMDB_URL,
            params={
                "apikey": OMDB_API_KEY,
                "i": imdb_id,
                "plot": "full"
            },
            timeout=4
        )


        if response.status_code == 200:

            data = response.json()

            if data.get("Response") == "True":

                return data

    except Exception:

        pass


    return None


# ============================================================
# POSTER
# ============================================================

def get_poster(title):

    details = get_movie_details(title)

    if not details:

        return None, None


    poster = details.get(
        "Poster"
    )


    if poster == "N/A":

        poster = None


    return poster, details


# ============================================================
# FAST POSTER LOADING
# ============================================================

def get_posters_fast(titles):

    results = {}


    with ThreadPoolExecutor(
        max_workers=8
    ) as executor:

        futures = {
            executor.submit(
                get_poster,
                title
            ): title
            for title in titles
        }


        for future in futures:

            title = futures[future]


            try:

                results[title] = (
                    future.result()
                )

            except Exception:

                results[title] = (
                    None,
                    None
                )


    return results


# ============================================================
# SEARCH MOVIES
# ============================================================

def search_movies(query):

    query = query.strip().lower()


    if not query:

        return []


    titles = movies[
        "title"
    ].tolist()


    exact = [
        title
        for title in titles
        if title.lower() == query
    ]


    starts = [
        title
        for title in titles
        if title.lower().startswith(query)
        and title not in exact
    ]


    contains = [
        title
        for title in titles
        if query in title.lower()
        and title not in exact
        and title not in starts
    ]


    return (
        exact
        + starts
        + contains
    )[:10]


# ============================================================
# FAST RECOMMENDATIONS
# ============================================================

@st.cache_data(
    max_entries=500,
    show_spinner=False
)
def get_recommendations(
    title,
    count=8
):

    matches = movies[
        movies["title"]
        .str.lower()
        .str.strip()
        ==
        title.lower().strip()
    ]


    if matches.empty:

        return []


    movie_index = matches.index[0]


    movie_vector = movie_vectors[
        movie_index
    ]


    scores = cosine_similarity(
        movie_vector,
        movie_vectors
    ).flatten()


    top_indexes = np.argsort(
        scores
    )[::-1]


    recommendations = []


    for index in top_indexes:

        if index == movie_index:

            continue


        movie_title = movies.iloc[
            index
        ]["title"]


        if not movie_title:

            continue


        recommendations.append(
            (
                movie_title,
                float(scores[index])
            )
        )


        if len(recommendations) >= count:

            break


    return recommendations


# ============================================================
# MOVIE GRID
# ============================================================

def show_movie_grid(
    movie_titles,
    hide_missing=True,
    favorites_mode=False
):

    if not movie_titles:

        return


    movie_titles = list(
        dict.fromkeys(
            movie_titles
        )
    )


    poster_data = get_posters_fast(
        movie_titles
    )


    valid_movies = []


    for title in movie_titles:

        poster, details = poster_data.get(
            title,
            (None, None)
        )


        if hide_missing and not poster:

            continue


        valid_movies.append(
            (
                title,
                poster,
                details
            )
        )


    if not valid_movies:

        st.info(
            "No movies available."
        )

        return


    # ========================================================
    # GRID
    # ========================================================

    for start in range(
        0,
        len(valid_movies),
        4
    ):

        row = valid_movies[
            start:start + 4
        ]


        columns = st.columns(4)


        for column, data in zip(
            columns,
            row
        ):

            title, poster, details = data


            with column:

                st.markdown(
                    '<div class="movie-card">',
                    unsafe_allow_html=True
                )


                # POSTER
                if poster:

                    st.image(
                        poster,
                        use_container_width=True
                    )

                else:

                    st.markdown(
                        """
                        <div style="
                        height:300px;
                        background:#333;
                        display:flex;
                        align-items:center;
                        justify-content:center;
                        border-radius:8px;
                        color:#aaa;">
                        🎬
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


                # TITLE
                st.markdown(
                    f"""
                    <div class="movie-title">
                    {title}
                    </div>
                    """,
                    unsafe_allow_html=True
                )


                # INFO
                if details:

                    rating = details.get(
                        "imdbRating",
                        "N/A"
                    )

                    year = details.get(
                        "Year",
                        "N/A"
                    )

                    st.caption(
                        f"⭐ {rating}  |  📅 {year}"
                    )


                # DETAILS BUTTON
                if st.button(
                    "View Details",
                    key=f"view_{title}_{start}_{favorites_mode}"
                ):

                    open_movie(title)


                # FAVORITE REMOVE
                if favorites_mode:

                    if st.button(
                        "❌ Remove",
                        key=f"remove_{title}_{start}"
                    ):

                        remove_favorite(title)

                        st.rerun()


                st.markdown(
                    "</div>",
                    unsafe_allow_html=True
                )


# ============================================================
# POPULAR MOVIES
# ============================================================

@st.cache_data
def get_popular_movies():

    data = movies.copy()


    C = data[
        "vote_average"
    ].mean()


    m = data[
        "vote_count"
    ].quantile(0.70)


    data = data[
        data["vote_count"] >= m
    ].copy()


    data["weighted"] = (
        (
            data["vote_count"]
            /
            (
                data["vote_count"]
                + m
            )
        )
        *
        data["vote_average"]
        +
        (
            m
            /
            (
                data["vote_count"]
                + m
            )
        )
        *
        C
    )


    data = data.sort_values(
        "weighted",
        ascending=False
    )


    return data[
        "title"
    ].head(8).tolist()


# ============================================================
# TRENDING
# ============================================================

@st.cache_data
def get_trending_movies():

    data = movies.sort_values(
        [
            "vote_count",
            "vote_average"
        ],
        ascending=False
    )


    return data[
        "title"
    ].head(8).tolist()


# ============================================================
# HOME PAGE
# ============================================================

def home_page():

    # ========================================================
    # HEADER
    # ========================================================

    st.markdown(
        """
        <div class="main-title">
        🎬 Netflix Movie Recommender
        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class="subtitle">
        Discover movies you love
        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # SEARCH
    # ========================================================

    search_query = st.text_input(
        "🔍 Search for a movie",
        placeholder="Search for the Movie...",
        key="home_search"
    )


    # ========================================================
    # SEARCH RESULTS
    # ========================================================

    if search_query.strip():

        results = search_movies(
            search_query
        )


        if not results:

            st.warning(
                "No movies found."
            )

            return


        # Exact match
        recommendation_movie = results[0]


        for movie in results:

            if movie.lower() == (
                search_query
                .strip()
                .lower()
            ):

                recommendation_movie = movie

                break


        st.session_state.recommendation_source = (
            recommendation_movie
        )


        # ----------------------------------------------------
        # SEARCH RESULTS
        # ----------------------------------------------------

        st.markdown(
            """
            <div class="section-title">
            🔎 Search Results
            </div>
            """,
            unsafe_allow_html=True
        )


        show_movie_grid(
            results[:4],
            hide_missing=True
        )


        # ----------------------------------------------------
        # RECOMMENDATIONS
        # ----------------------------------------------------

        st.markdown("---")


        st.markdown(
            f"""
            <div class="section-title">
            🎯 Recommended for "{recommendation_movie}"
            </div>
            """,
            unsafe_allow_html=True
        )


        recommendations = get_recommendations(
            recommendation_movie,
            8
        )


        recommendation_titles = [
            movie
            for movie, score
            in recommendations
        ]


        show_movie_grid(
            recommendation_titles,
            hide_missing=True
        )


        return


    # ========================================================
    # PERSONALIZED RECOMMENDATIONS
    # ========================================================

    source = (
        st.session_state.recommendation_source
    )


    if source:

        recommendations = get_recommendations(
            source,
            8
        )


        if recommendations:

            st.markdown(
                f"""
                <div class="section-title">
                🎯 Because You Watched "{source}"
                </div>
                """,
                unsafe_allow_html=True
            )


            titles = [
                movie
                for movie, score
                in recommendations
            ]


            show_movie_grid(
                titles,
                hide_missing=True
            )


    # ========================================================
    # POPULAR
    # ========================================================

    st.markdown(
        """
        <div class="section-title">
        🔥 Popular Movies
        </div>
        """,
        unsafe_allow_html=True
    )


    show_movie_grid(
        get_popular_movies(),
        hide_missing=True
    )


    # ========================================================
    # TRENDING
    # ========================================================

    st.markdown(
        """
        <div class="section-title">
        📈 Trending Movies
        </div>
        """,
        unsafe_allow_html=True
    )


    show_movie_grid(
        get_trending_movies(),
        hide_missing=True
    )


# ============================================================
# FAVORITES PAGE
# ============================================================

def favorites_page():

    st.markdown(
        """
        <div class="main-title">
        ❤️ My Favourites
        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class="subtitle">
        Movies you saved to watch later
        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # EMPTY
    # ========================================================

    if not st.session_state.favorites:

        st.markdown(
            """
            <div style="
            background:#1f1f1f;
            padding:50px;
            border-radius:12px;
            text-align:center;
            margin-top:30px;
            ">

            <div style="font-size:55px;">
            ❤️
            </div>

            <h2 style="color:white;">
            Your favourites are empty
            </h2>

            <p style="color:#aaa;">
            Add movies to your favourites
            and they will appear here.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

        return


    # ========================================================
    # FAVOURITE COUNT
    # ========================================================

    st.caption(
        f"{len(st.session_state.favorites)} "
        "movie(s) saved"
    )


    # ========================================================
    # MOVIES
    # ========================================================

    show_movie_grid(
        st.session_state.favorites,
        hide_missing=False,
        favorites_mode=True
    )


# ============================================================
# DETAILS PAGE
# ============================================================

def movie_details_page():

    title = st.session_state.selected_movie


    if not title:

        st.session_state.page = "Home"

        st.rerun()


    # ========================================================
    # BACK
    # ========================================================

    if st.button("← Back to Home"):

        st.session_state.page = "Home"

        st.rerun()


    # ========================================================
    # DETAILS
    # ========================================================

    details = get_movie_details(
        title
    )


    if not details:

        st.error(
            "Movie details could not be loaded."
        )

        return


    poster = details.get(
        "Poster"
    )


    col1, col2 = st.columns(
        [1, 2]
    )


    with col1:

        if poster and poster != "N/A":

            st.image(
                poster,
                use_container_width=True
            )


    with col2:

        st.markdown(
            f"""
            <h1 style="color:white;">
            {details.get("Title", title)}
            </h1>
            """,
            unsafe_allow_html=True
        )


        st.write(
            f"📅 **Year:** "
            f"{details.get('Year', 'N/A')}"
        )


        st.write(
            f"⭐ **IMDb Rating:** "
            f"{details.get('imdbRating', 'N/A')}"
        )


        st.write(
            f"🎭 **Genre:** "
            f"{details.get('Genre', 'N/A')}"
        )


        st.write(
            f"🎬 **Director:** "
            f"{details.get('Director', 'N/A')}"
        )


        st.write(
            f"👥 **Actors:** "
            f"{details.get('Actors', 'N/A')}"
        )


        st.write(
            f"⏱️ **Runtime:** "
            f"{details.get('Runtime', 'N/A')}"
        )


        st.markdown(
            "### 📝 Plot"
        )


        st.write(
            details.get(
                "Plot",
                "No plot available."
            )
        )


        # ====================================================
        # FAVORITE BUTTON
        # ====================================================

        st.markdown("---")


        if is_favorite(title):

            if st.button(
                "❤️ Remove from Favourites"
            ):

                remove_favorite(title)

                st.rerun()

        else:

            if st.button(
                "🤍 Add to Favourites"
            ):

                add_favorite(title)

                st.rerun()


    # ========================================================
    # RECOMMENDATIONS
    # ========================================================

    st.markdown("---")


    st.markdown(
        """
        <div class="section-title">
        🎯 You May Also Like
        </div>
        """,
        unsafe_allow_html=True
    )


    recommendations = get_recommendations(
        title,
        8
    )


    titles = [
        movie
        for movie, score
        in recommendations
    ]


    show_movie_grid(
        titles,
        hide_missing=True
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="logo">
        🎬 NETFLIX
        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # HOME
    # ========================================================

    if st.button(
        "🏠  Home",
        key="sidebar_home"
    ):

        st.session_state.page = "Home"

        st.rerun()


    # ========================================================
    # FAVORITES
    # ========================================================

    favorite_count = len(
        st.session_state.favorites
    )


    if st.button(
        f"❤️  Favourites  ({favorite_count})",
        key="sidebar_favorites"
    ):

        st.session_state.page = "Favorites"

        st.rerun()


   


    

# ============================================================
# PAGE ROUTING
# ============================================================

if st.session_state.page == "Home":

    home_page()


elif st.session_state.page == "Favorites":

    favorites_page()


elif st.session_state.page == "Details":

    movie_details_page()