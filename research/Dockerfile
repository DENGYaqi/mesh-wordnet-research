FROM public.ecr.aws/docker/library/python:3.7.17-slim-bullseye
WORKDIR /research
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt
