# name=Dockerfile
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UPLOAD_DIR=/tmp/uploads

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy entire repo (adjust if needed)
COPY . /app

RUN mkdir -p ${UPLOAD_DIR} && chmod 777 ${UPLOAD_DIR}

EXPOSE 5000
CMD ["gunicorn", "server.app:app", "-b", "0.0.0.0:5000", "--timeout", "300"]
