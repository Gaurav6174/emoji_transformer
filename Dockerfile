#  Base image 
FROM python:3.10-slim

#  System dependencies 
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

#  Hugging Face Spaces runs containers as UID 1000 
RUN useradd -m -u 1000 appuser

#  Working directory 
WORKDIR /app

#  Install Python dependencies first
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

#  Copy project files 
COPY --chown=appuser:appuser . .

#  Streamlit config — disable telemetry, set server options 
RUN mkdir -p /app/.streamlit
COPY --chown=appuser:appuser .streamlit/config.toml /app/.streamlit/config.toml

#  Switch to non-root user 
USER appuser

#  Expose Hugging Face Spaces default port 
EXPOSE 7860

#  Launch Streamlit 
CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
