## Deployment

### Local development

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py  # runs on http://localhost:5000
```

### Production tips

- Use Gunicorn or uWSGI instead of the Flask development server.
- Set `FLASK_ENV=production` or avoid debug mode.
- Precompute models and ensure `models/` artifacts exist before serving traffic.
- Serve static files (CSS, images) via a CDN or Nginx for performance.

### Gunicorn example

```bash
pip install gunicorn
# Bind to 0.0.0.0 for containers; adjust workers based on CPU
gunicorn -w 2 -b 0.0.0.0:5000 app:app
```

### Docker (sketch)

Create `Dockerfile`:
```Dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Pre-warm models here if needed (or mount at runtime)
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5000", "app:app"]
```

Build and run:
```bash
docker build -t smart-urban .
docker run -p 5000:5000 smart-urban
```

### Environment variables

- None mandatory. Add secrets/config via env vars if you introduce external services (databases, APIs).