## Frontend Documentation

The frontend is a minimal server-rendered UI using Flask templates (Jinja2) and vanilla HTML/CSS.

### Templates

- `templates/dashboard.html`
  - Displays a header, basic nav, and three panels:
    - Air Quality Prediction form → posts to `/predict/air_quality`
    - Traffic Congestion Prediction form → posts to `/predict/traffic`
    - Affordable Housing Prediction form → posts to `/predict/housing`
  - After submission, the same template is re-rendered with a prediction paragraph and an image preview (plots saved during training).

- `templates/index.html`
  - Simple landing page (not actively routed in `app.py`). You can add a route to serve it if needed.

### Forms and fields

- Air Quality:
  - `Country` (text)
  - `Status` (text)
  - `year` (number)
  - `month` (number)
  - `day` (number)

- Traffic:
  - `hour` (number, 0–23)
  - `day_of_week` (number, 0–6)

- Housing:
  - `population_density` (number)
  - `average_income` (number)

### Styling

- `static/css/style.css` is included but largely empty; `dashboard.html` contains inline styles to ensure a clean look.
- To centralize styling, move the inline `<style>` from the template into `static/css/style.css` and update the `<link>` tag.

### Extending the UI

- Add client-side validation to ensure numeric ranges (e.g., month 1–12) and better UX.
- For a richer experience, convert to a SPA (React/Vue/Svelte) that calls the same Flask endpoints and displays JSON results.
- Add charts: Use the saved plots (`static/images/...`) or generate JSON and render with Chart.js/D3 on the client.

### Navigation and layout

- Header nav contains a single Home link pointing to `/` which renders `dashboard.html` by default.
- Add additional pages by creating new templates and routes in `app.py`.