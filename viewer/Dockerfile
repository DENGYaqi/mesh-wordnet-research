FROM public.ecr.aws/docker/library/python:3.11.13-slim-bookworm
WORKDIR /viewer
ENV TRANSLATION_MODEL_DIR=/opt/translate-en_zh-1_9
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY install_translation_model.py ./
RUN python install_translation_model.py
COPY app.py results.py test_results.py ./
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--browser.gatherUsageStats=false", "--client.toolbarMode=minimal"]
