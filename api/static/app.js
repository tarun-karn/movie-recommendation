const API_BASE = window.location.origin; // Assume API is served from same origin

// =========================
// DOM ELEMENTS
// =========================
const mainContent = document.getElementById('main-content');
const searchForm = document.getElementById('search-form');
const searchInput = document.getElementById('search-input');

// =========================
// ROUTER
// =========================
window.addEventListener('hashchange', handleRoute);
window.addEventListener('DOMContentLoaded', handleRoute);

function handleRoute() {
  const hash = window.location.hash || '#/';
  
  if (hash === '#/') {
    renderHome();
  } else if (hash.startsWith('#/search/')) {
    const query = decodeURIComponent(hash.replace('#/search/', ''));
    renderSearch(query);
  } else if (hash.startsWith('#/movie/')) {
    const title = decodeURIComponent(hash.replace('#/movie/', ''));
    renderMovieDetail(title);
  } else {
    renderHome();
  }
}

// =========================
// EVENT LISTENERS
// =========================
searchForm.addEventListener('submit', (e) => {
  e.preventDefault();
  const query = searchInput.value.trim();
  if (query) {
    window.location.hash = `#/search/${encodeURIComponent(query)}`;
  }
});

// =========================
// RENDERERS
// =========================
function showLoading(msg = 'Loading...') {
  mainContent.innerHTML = `<div class="container"><div class="message-box loading">${msg}</div></div>`;
}

function showError(msg) {
  mainContent.innerHTML = `<div class="container"><div class="message-box">${msg}</div></div>`;
}

function createMovieCard(movie) {
  const year = movie.release_date ? movie.release_date.split('-')[0] : 'N/A';
  const rating = movie.vote_average ? movie.vote_average.toFixed(1) : 'NR';
  const poster = movie.poster_url || 'https://via.placeholder.com/500x750?text=No+Poster';
  
  return `
    <article class="movie-card" onclick="window.location.hash='#/movie/${encodeURIComponent(movie.title)}'">
      <img src="${poster}" alt="${movie.title}" class="movie-poster" loading="lazy">
      <div class="movie-info">
        <h3 class="movie-title">${movie.title}</h3>
        <div class="movie-meta">
          <span>${year}</span>
          <span class="rating">⭐ ${rating}</span>
        </div>
      </div>
    </article>
  `;
}

function createGrid(movies) {
  if (!movies || movies.length === 0) return '<p>No movies found.</p>';
  return `<div class="movie-grid">${movies.map(createMovieCard).join('')}</div>`;
}

function createScroller(movies) {
  if (!movies || movies.length === 0) return '<p>No recommendations available.</p>';
  return `<div class="scroller-container"><div class="scroller">${movies.map(createMovieCard).join('')}</div></div>`;
}

async function renderHome() {
  showLoading('Loading popular movies...');
  try {
    const res = await fetch(`${API_BASE}/home?category=popular&limit=24`);
    if (!res.ok) throw new Error('Failed to fetch home feed');
    const movies = await res.json();
    
    mainContent.innerHTML = `
      <div class="container">
        <h2 class="section-title">Trending Popular Movies</h2>
        ${createGrid(movies)}
      </div>
    `;
  } catch (err) {
    showError(err.message);
  }
}

async function renderSearch(query) {
  showLoading(`Searching TMDB for "${query}"...`);
  searchInput.value = query;
  try {
    const res = await fetch(`${API_BASE}/tmdb/search?query=${encodeURIComponent(query)}`);
    if (!res.ok) throw new Error('Search failed');
    const data = await res.json();
    
    // Map TMDB raw results to our card format
    const movies = data.results.map(m => ({
      tmdb_id: m.id,
      title: m.title || m.name,
      poster_url: m.poster_path ? `https://image.tmdb.org/t/p/w500${m.poster_path}` : null,
      release_date: m.release_date,
      vote_average: m.vote_average
    }));

    mainContent.innerHTML = `
      <div class="container">
        <h2 class="section-title">Search Results: ${query}</h2>
        ${createGrid(movies)}
      </div>
    `;
  } catch (err) {
    showError(err.message);
  }
}

async function renderMovieDetail(title) {
  showLoading(`Loading details and AI recommendations for "${title}"...`);
  try {
    const res = await fetch(`${API_BASE}/movie/search?query=${encodeURIComponent(title)}&tfidf_top_n=10&genre_limit=10`);
    if (!res.ok) {
      if (res.status === 404) throw new Error(`Movie "${title}" not found.`);
      throw new Error('Failed to load movie details');
    }
    const data = await res.json();
    const movie = data.movie_details;
    
    // Prepare TF-IDF Recs (map from TFIDFRecItem to TMDBMovieCard for UI consistency)
    const tfidfRecs = data.tfidf_recommendations.map(r => r.tmdb || { title: r.title });
    
    // Genres
    const genres = movie.genres.map(g => `<span class="genre-tag">${g.name}</span>`).join('');
    const year = movie.release_date ? movie.release_date.split('-')[0] : '';
    const poster = movie.poster_url || 'https://via.placeholder.com/500x750?text=No+Poster';

    mainContent.innerHTML = `
      <div class="container detail-view">
        <div class="detail-header brutal-box">
          <img src="${poster}" alt="${movie.title}" class="detail-poster">
          <div class="detail-info">
            <h1 class="detail-title">${movie.title}</h1>
            <div class="detail-meta">
              <span>${year}</span>
            </div>
            <div class="genre-tags">${genres}</div>
            <p class="detail-overview" style="margin-top: 20px;">${movie.overview || 'No overview available.'}</p>
          </div>
        </div>

        <div>
          <h2 class="section-title">AI Similar Movies (Content-Based)</h2>
          ${createScroller(tfidfRecs.filter(m => m.poster_url))} 
        </div>

        <div>
          <h2 class="section-title">More in this Genre</h2>
          ${createScroller(data.genre_recommendations)}
        </div>
      </div>
    `;
  } catch (err) {
    showError(err.message);
  }
}
