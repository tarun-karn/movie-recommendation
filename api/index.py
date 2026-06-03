import os
import logging
import pickle
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import httpx
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
# pyrefly: ignore [missing-import]
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv


# =========================
# LOGGING
# =========================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =========================
# ENV
# =========================
load_dotenv()

TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")
TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMG_500 = "https://image.tmdb.org/t/p/w500"

# Configurable CORS origins (comma-separated in env, defaults to all for dev)
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

# Environment mode
IS_PRODUCTION = os.getenv("VERCEL_ENV", "development") == "production"


# =========================
# GLOBAL STATE
# =========================
_state: Dict[str, Any] = {
    "df": None,
    "tfidf_matrix": None,
    "title_to_idx": None,
    "loaded": False,
}


# =========================
# DATA LOADING
# =========================
def _resolve_data_path(filename: str) -> str:
    """Resolve data file paths that work both locally and on Vercel."""
    # Data is now co-located with api/index.py in api/data
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "data", filename)


def _norm_title(t: str) -> str:
    return str(t).strip().lower()


def _build_title_to_idx(indices: Any) -> Dict[str, int]:
    """
    Normalize indices.pkl into a {lowercase_title: row_index} dict.
    Handles both dict and pandas Series formats.
    """
    title_to_idx: Dict[str, int] = {}
    try:
        for k, v in indices.items():
            title_to_idx[_norm_title(k)] = int(v)
    except Exception as exc:
        raise RuntimeError(
            "indices.pkl must be dict or pandas Series-like (with .items())"
        ) from exc
    return title_to_idx


def load_data() -> None:
    """Load pickle data files into global state. Idempotent."""
    if _state["loaded"]:
        return

    logger.info("Loading pickle data files...")

    paths = {
        "df": _resolve_data_path("df.pkl"),
        "indices": _resolve_data_path("indices.pkl"),
        "tfidf_matrix": _resolve_data_path("tfidf_matrix.pkl"),
    }

    for name, path in paths.items():
        if not os.path.exists(path):
            raise RuntimeError(f"Required data file not found: {path}")

    with open(paths["df"], "rb") as f:
        _state["df"] = pickle.load(f)

    with open(paths["indices"], "rb") as f:
        indices_obj = pickle.load(f)

    with open(paths["tfidf_matrix"], "rb") as f:
        _state["tfidf_matrix"] = pickle.load(f)

    _state["title_to_idx"] = _build_title_to_idx(indices_obj)

    # Sanity check
    if _state["df"] is None or "title" not in _state["df"].columns:
        raise RuntimeError("df.pkl must contain a DataFrame with a 'title' column")

    _state["loaded"] = True
    logger.info(
        "Data loaded: %d movies, %d index entries",
        len(_state["df"]),
        len(_state["title_to_idx"]),
    )


# =========================
# LIFESPAN (replaces deprecated on_event)
# =========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load data on startup, cleanup on shutdown."""
    if not TMDB_API_KEY:
        logger.warning(
            "TMDB_API_KEY is not set. TMDB-dependent endpoints will fail."
        )
    load_data()
    yield
    logger.info("Shutting down Movie Recommender API")


# =========================
# FASTAPI APP
# =========================
app = FastAPI(
    title="Movie Recommender API",
    description="Content-based movie recommendations with TF-IDF & TMDB integration",
    version="3.0.0",
    lifespan=lifespan,
    docs_url="/docs" if not IS_PRODUCTION else None,
    redoc_url="/redoc" if not IS_PRODUCTION else None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["*"],
)


# =========================
# SECURITY MIDDLEWARE
# =========================
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response


# =========================
# GLOBAL ERROR HANDLER
# =========================
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled error on %s: %s", request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error"
            if IS_PRODUCTION
            else str(exc)
        },
    )


# =========================
# MODELS
# =========================
class TMDBMovieCard(BaseModel):
    tmdb_id: int
    title: str
    poster_url: Optional[str] = None
    release_date: Optional[str] = None
    vote_average: Optional[float] = None


class TMDBMovieDetails(BaseModel):
    tmdb_id: int
    title: str
    overview: Optional[str] = None
    release_date: Optional[str] = None
    poster_url: Optional[str] = None
    backdrop_url: Optional[str] = None
    genres: List[dict] = Field(default_factory=list)


class TFIDFRecItem(BaseModel):
    title: str
    score: float
    tmdb: Optional[TMDBMovieCard] = None


class SearchBundleResponse(BaseModel):
    query: str
    movie_details: TMDBMovieDetails
    tfidf_recommendations: List[TFIDFRecItem]
    genre_recommendations: List[TMDBMovieCard]


# =========================
# TMDB HELPERS
# =========================
def _require_tmdb_key() -> None:
    if not TMDB_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="TMDB API key is not configured. Set TMDB_API_KEY env var.",
        )


def make_img_url(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    return f"{TMDB_IMG_500}{path}"


async def tmdb_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Safe TMDB API GET request.
    Raises HTTPException on network/API errors.
    """
    _require_tmdb_key()
    q = dict(params)
    q["api_key"] = TMDB_API_KEY

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(f"{TMDB_BASE}{path}", params=q)
    except httpx.RequestError as e:
        logger.error("TMDB request error: %s", e)
        raise HTTPException(
            status_code=502,
            detail="Failed to reach TMDB API. Please try again.",
        )

    if r.status_code != 200:
        logger.warning("TMDB API error %d on %s", r.status_code, path)
        raise HTTPException(
            status_code=502,
            detail=f"TMDB API returned status {r.status_code}",
        )

    return r.json()


async def tmdb_cards_from_results(
    results: List[dict], limit: int = 20
) -> List[TMDBMovieCard]:
    out: List[TMDBMovieCard] = []
    for m in (results or [])[:limit]:
        mid = m.get("id")
        if not mid:
            continue
        out.append(
            TMDBMovieCard(
                tmdb_id=int(mid),
                title=m.get("title") or m.get("name") or "",
                poster_url=make_img_url(m.get("poster_path")),
                release_date=m.get("release_date"),
                vote_average=m.get("vote_average"),
            )
        )
    return out


async def tmdb_movie_details(movie_id: int) -> TMDBMovieDetails:
    data = await tmdb_get(f"/movie/{movie_id}", {"language": "en-US"})
    return TMDBMovieDetails(
        tmdb_id=int(data["id"]),
        title=data.get("title") or "",
        overview=data.get("overview"),
        release_date=data.get("release_date"),
        poster_url=make_img_url(data.get("poster_path")),
        backdrop_url=make_img_url(data.get("backdrop_path")),
        genres=data.get("genres") or [],
    )


async def tmdb_search_movies(query: str, page: int = 1) -> Dict[str, Any]:
    """Raw TMDB search response (multiple results)."""
    return await tmdb_get(
        "/search/movie",
        {
            "query": query,
            "include_adult": "false",
            "language": "en-US",
            "page": page,
        },
    )


async def tmdb_search_first(query: str) -> Optional[dict]:
    data = await tmdb_search_movies(query=query, page=1)
    results = data.get("results", [])
    return results[0] if results else None


# =========================
# TF-IDF HELPERS
# =========================
def _get_local_idx(title: str) -> int:
    title_to_idx = _state["title_to_idx"]
    if title_to_idx is None:
        raise HTTPException(status_code=500, detail="TF-IDF index not initialized")
    key = _norm_title(title)
    if key in title_to_idx:
        return int(title_to_idx[key])
    raise HTTPException(
        status_code=404, detail=f"Title not found in local dataset: '{title}'"
    )


def tfidf_recommend_titles(
    query_title: str, top_n: int = 10
) -> List[Tuple[str, float]]:
    """
    Compute TF-IDF cosine similarity recommendations.
    Returns list of (title, similarity_score).
    """
    df = _state["df"]
    tfidf_matrix = _state["tfidf_matrix"]
    if df is None or tfidf_matrix is None:
        raise HTTPException(status_code=500, detail="TF-IDF resources not loaded")

    idx = _get_local_idx(query_title)

    # Cosine similarity via sparse matrix multiplication
    qv = tfidf_matrix[idx]
    scores = (tfidf_matrix @ qv.T).toarray().ravel()
    order = np.argsort(-scores)

    out: List[Tuple[str, float]] = []
    for i in order:
        if int(i) == int(idx):
            continue
        try:
            title_i = str(df.iloc[int(i)]["title"])
        except (IndexError, KeyError):
            continue
        out.append((title_i, float(scores[int(i)])))
        if len(out) >= top_n:
            break
    return out


async def attach_tmdb_card(title: str) -> Optional[TMDBMovieCard]:
    """Fetch TMDB card for a local title. Returns None on failure."""
    try:
        m = await tmdb_search_first(title)
        if not m:
            return None
        return TMDBMovieCard(
            tmdb_id=int(m["id"]),
            title=m.get("title") or title,
            poster_url=make_img_url(m.get("poster_path")),
            release_date=m.get("release_date"),
            vote_average=m.get("vote_average"),
        )
    except Exception:
        return None


# =========================
# ROUTES
# =========================


@app.get("/health", tags=["Health"])
def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "data_loaded": _state["loaded"],
        "movie_count": len(_state["df"]) if _state["df"] is not None else 0,
    }


# ---------- HOME FEED (TMDB) ----------
@app.get("/home", response_model=List[TMDBMovieCard], tags=["Movies"])
async def home(
    category: str = Query(
        "popular",
        description="Feed category",
        pattern="^(trending|popular|top_rated|upcoming|now_playing)$",
    ),
    limit: int = Query(24, ge=1, le=50, description="Number of results"),
):
    """Home feed — browse movies by category from TMDB."""
    try:
        if category == "trending":
            data = await tmdb_get("/trending/movie/day", {"language": "en-US"})
        else:
            data = await tmdb_get(
                f"/movie/{category}", {"language": "en-US", "page": 1}
            )
        return await tmdb_cards_from_results(data.get("results", []), limit=limit)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Home route failed: %s", e)
        raise HTTPException(status_code=500, detail="Failed to load home feed")


# ---------- TMDB KEYWORD SEARCH ----------
@app.get("/tmdb/search", tags=["Search"])
async def tmdb_search(
    query: str = Query(
        ..., min_length=1, max_length=200, description="Movie title keyword"
    ),
    page: int = Query(1, ge=1, le=10, description="Result page"),
):
    """Search TMDB for movies by keyword. Returns raw TMDB result shape."""
    return await tmdb_search_movies(query=query, page=page)


# ---------- MOVIE DETAILS ----------
@app.get(
    "/movie/id/{tmdb_id}",
    response_model=TMDBMovieDetails,
    tags=["Movies"],
)
async def movie_details_route(tmdb_id: int):
    """Get detailed movie information from TMDB."""
    if tmdb_id <= 0:
        raise HTTPException(status_code=400, detail="Invalid movie ID")
    return await tmdb_movie_details(tmdb_id)


# ---------- GENRE RECOMMENDATIONS ----------
@app.get(
    "/recommend/genre",
    response_model=List[TMDBMovieCard],
    tags=["Recommendations"],
)
async def recommend_genre(
    tmdb_id: int = Query(..., gt=0, description="TMDB movie ID"),
    limit: int = Query(18, ge=1, le=50, description="Number of results"),
):
    """Recommend movies in the same genre as the given movie."""
    details = await tmdb_movie_details(tmdb_id)
    if not details.genres:
        return []

    genre_id = details.genres[0]["id"]
    discover = await tmdb_get(
        "/discover/movie",
        {
            "with_genres": genre_id,
            "language": "en-US",
            "sort_by": "popularity.desc",
            "page": 1,
        },
    )
    cards = await tmdb_cards_from_results(discover.get("results", []), limit=limit)
    return [c for c in cards if c.tmdb_id != tmdb_id]


# ---------- TF-IDF RECOMMENDATIONS ----------
@app.get("/recommend/tfidf", tags=["Recommendations"])
async def recommend_tfidf(
    title: str = Query(
        ..., min_length=1, max_length=200, description="Movie title"
    ),
    top_n: int = Query(10, ge=1, le=50, description="Number of results"),
):
    """Get content-based recommendations using TF-IDF similarity."""
    recs = tfidf_recommend_titles(title, top_n=top_n)
    return [{"title": t, "score": round(s, 4)} for t, s in recs]


# ---------- BUNDLE: Details + TF-IDF + Genre ----------
@app.get(
    "/movie/search",
    response_model=SearchBundleResponse,
    tags=["Search"],
)
async def search_bundle(
    query: str = Query(
        ..., min_length=1, max_length=200, description="Movie title"
    ),
    tfidf_top_n: int = Query(12, ge=1, le=30),
    genre_limit: int = Query(12, ge=1, le=30),
):
    """
    All-in-one search: movie details + TF-IDF recommendations + genre recommendations.
    Best for when a user selects a specific movie.
    """
    best = await tmdb_search_first(query)
    if not best:
        raise HTTPException(
            status_code=404, detail=f"No TMDB movie found for: {query}"
        )

    tmdb_id = int(best["id"])
    details = await tmdb_movie_details(tmdb_id)

    # TF-IDF recommendations (graceful fallback)
    tfidf_items: List[TFIDFRecItem] = []
    recs: List[Tuple[str, float]] = []
    try:
        recs = tfidf_recommend_titles(details.title, top_n=tfidf_top_n)
    except Exception:
        try:
            recs = tfidf_recommend_titles(query, top_n=tfidf_top_n)
        except Exception:
            recs = []

    for title, score in recs:
        card = await attach_tmdb_card(title)
        tfidf_items.append(TFIDFRecItem(title=title, score=round(score, 4), tmdb=card))

    # Genre recommendations
    genre_recs: List[TMDBMovieCard] = []
    if details.genres:
        genre_id = details.genres[0]["id"]
        discover = await tmdb_get(
            "/discover/movie",
            {
                "with_genres": genre_id,
                "language": "en-US",
                "sort_by": "popularity.desc",
                "page": 1,
            },
        )
        cards = await tmdb_cards_from_results(
            discover.get("results", []), limit=genre_limit
        )
        genre_recs = [c for c in cards if c.tmdb_id != details.tmdb_id]

    return SearchBundleResponse(
        query=query,
        movie_details=details,
        tfidf_recommendations=tfidf_items,
        genre_recommendations=genre_recs,
    )


# =========================
# STATIC FRONTEND
# =========================
# Frontend files live in api/static/ (co-located with this file).
# On Vercel, subdirectories of api/ are auto-bundled with the function.
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


@app.get("/", tags=["Frontend"], include_in_schema=False)
def serve_index():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type="text/html")
    return JSONResponse(
        status_code=404,
        content={"detail": f"Frontend not found at {index_path}"},
    )


if os.path.exists(static_dir):
    app.mount("/", StaticFiles(directory=static_dir), name="static")

