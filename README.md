# 🎬 Movie Recommender API

A production-ready, content-based movie recommendation engine deployed as a serverless API on Vercel. Uses **TF-IDF cosine similarity** on a 45,000+ movie dataset combined with real-time **TMDB API** integration for rich movie data, posters, and genre-based discovery.

[![CI](https://github.com/tarun-karn/movie-recommendation/actions/workflows/ci.yml/badge.svg)](https://github.com/tarun-karn/movie-recommendation/actions)
[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/import/project?template=https://github.com/tarun-karn/movie-recommendation)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://python.org)

---

## ✨ Features

- **Content-Based Recommendations** — TF-IDF vectorization + cosine similarity across 45,000+ movies
- **Real-Time Movie Data** — Live integration with TMDB API for posters, details, and metadata
- **Genre Discovery** — Browse and discover movies by genre, popularity, trending, and more
- **Bundle Endpoint** — Single API call returns movie details + TF-IDF recs + genre recs
- **Serverless Architecture** — Zero-config deployment on Vercel with automatic scaling
- **Production Security** — CORS configuration, security headers, input validation, sanitized errors
- **Interactive API Docs** — Auto-generated Swagger UI at `/docs` (development mode)

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Runtime** | Python 3.11 |
| **Framework** | FastAPI |
| **ML Engine** | TF-IDF (scikit-learn) + SciPy sparse matrices |
| **External API** | TMDB (The Movie Database) API v3 |
| **HTTP Client** | httpx (async) |
| **Data** | pandas, NumPy |
| **Deployment** | Vercel Serverless Functions |
| **CI/CD** | GitHub Actions |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Vercel Edge Network                   │
│                                                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │              FastAPI Serverless Function           │  │
│  │                   (api/index.py)                   │  │
│  │                                                   │  │
│  │  ┌─────────────┐    ┌──────────────────────────┐  │  │
│  │  │  TF-IDF     │    │    TMDB API Integration  │  │  │
│  │  │  Engine     │    │                          │  │  │
│  │  │             │    │  • Search movies         │  │  │
│  │  │  • Cosine   │    │  • Movie details         │  │  │
│  │  │    similarity│   │  • Genre discovery       │  │  │
│  │  │  • 45K+     │    │  • Trending / Popular    │  │  │
│  │  │    movies   │    │  • Poster URLs           │  │  │
│  │  └──────┬──────┘    └────────────┬─────────────┘  │  │
│  │         │                        │                │  │
│  │         ▼                        ▼                │  │
│  │  ┌─────────────┐    ┌──────────────────────────┐  │  │
│  │  │  Pickle     │    │   TMDB API v3            │  │  │
│  │  │  Data Files │    │   (api.themoviedb.org)   │  │  │
│  │  │  (data/)    │    │                          │  │  │
│  │  └─────────────┘    └──────────────────────────┘  │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API info and status |
| `GET` | `/health` | Health check with data loading status |
| `GET` | `/home?category=popular&limit=24` | Home feed (trending, popular, top_rated, upcoming, now_playing) |
| `GET` | `/tmdb/search?query=batman&page=1` | Search TMDB by keyword |
| `GET` | `/movie/id/{tmdb_id}` | Full movie details from TMDB |
| `GET` | `/recommend/tfidf?title=Inception&top_n=10` | Content-based TF-IDF recommendations |
| `GET` | `/recommend/genre?tmdb_id=27205&limit=18` | Genre-based recommendations |
| `GET` | `/movie/search?query=Inception` | Bundle: details + TF-IDF + genre recs |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- [TMDB API Key](https://www.themoviedb.org/settings/api) (free)

### Installation

```bash
# Clone the repository
git clone https://github.com/tarun-karn/movie-recommendation.git
cd movie-recommendation

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your TMDB_API_KEY
```

### Local Development

```bash
# Start the development server
uvicorn api.index:app --reload --port 8000

# The API is now running at http://localhost:8000
# Interactive docs at http://localhost:8000/docs
```

---

## 🔐 Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `TMDB_API_KEY` | ✅ Yes | — | Your TMDB API key ([get one here](https://www.themoviedb.org/settings/api)) |
| `CORS_ORIGINS` | No | `*` | Comma-separated allowed origins |
| `VERCEL_ENV` | No | `development` | Set automatically by Vercel |

---


## 📂 Project Structure

```
movie-recommendation/
├── api/
│   └── index.py              # FastAPI serverless function (main backend)
├── data/
│   ├── df.pkl                 # Pre-computed movie DataFrame (~30MB)
│   ├── indices.pkl            # Title-to-index mapping (~1.4MB)
│   ├── tfidf.pkl              # TF-IDF vectorizer object (~2MB)
│   └── tfidf_matrix.pkl       # Sparse TF-IDF similarity matrix (~19MB)
├── .github/
│   └── workflows/
│       └── ci.yml             # GitHub Actions CI pipeline
├── .env.example               # Environment variable template
├── .gitignore                 # Git ignore rules
├── LICENSE                    # MIT License
├── README.md                  # This file
├── requirements.txt           # Python dependencies
└── vercel.json                # Vercel deployment configuration
```

---

## 🔮 Future Improvements

- [ ] **Collaborative Filtering** — Add user-based recommendations using matrix factorization
- [ ] **Frontend Application** — Build a React/Next.js UI to consume the API
- [ ] **Caching Layer** — Add Redis caching for TMDB API responses to reduce latency
- [ ] **Database Migration** — Move from pickle files to PostgreSQL/SQLite for better scalability
- [ ] **User Accounts** — Authentication and personalized watchlists
- [ ] **Advanced Search** — Filter by year, rating, language, and cast
- [ ] **Rate Limiting** — Add per-IP rate limiting for public deployment
- [ ] **Monitoring** — Integrate with Sentry for error tracking and performance monitoring

---

## 👤 Author

**Tarun Karn**

- GitHub: [@tarun-karn](https://github.com/tarun-karn)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<p align="center">
  Built with ❤️ using FastAPI & TMDB API
</p>
