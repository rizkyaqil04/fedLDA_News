FROM python:3.11

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY config.yaml ./config.yaml
COPY client/ .


CMD ["python", "main.py"]

