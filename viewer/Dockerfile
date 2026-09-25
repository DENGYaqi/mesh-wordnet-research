FROM public.ecr.aws/docker/library/python:3.11.13-slim-bookworm
WORKDIR /viewer
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY app.py results.py test_results.py ./
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--browser.gatherUsageStats=false", "--client.toolbarMode=minimal"]
