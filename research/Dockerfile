FROM public.ecr.aws/docker/library/python:3.12.0-slim-bookworm@sha256:19a6235339a74eca01227b03629f63b6f5020abc21142436eced6ec3a9839a76
WORKDIR /research
COPY requirements.lock .
RUN python -m pip install --no-cache-dir -r requirements.lock && python -m pip check
ENV NLTK_DATA=/usr/local/nltk_data
RUN python -m nltk.downloader -d /usr/local/nltk_data punkt_tab \
    && echo 'e57f64187974277726a3417ca6f181ec5403676c717672eef6a748a7b20e0106  /usr/local/nltk_data/tokenizers/punkt_tab.zip' | sha256sum -c -
