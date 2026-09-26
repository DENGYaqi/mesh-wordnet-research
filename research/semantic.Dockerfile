FROM public.ecr.aws/docker/library/python:3.12.0-slim-bookworm@sha256:19a6235339a74eca01227b03629f63b6f5020abc21142436eced6ec3a9839a76

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /research
COPY semantic-requirements.lock /tmp/semantic-requirements.lock
RUN pip install --no-cache-dir -r /tmp/semantic-requirements.lock
