FROM python:3.10-slim

COPY . /app

ENV PYTHONPATH=/app

WORKDIR /app/src/

COPY requirements.txt .

COPY ./src/model.joblib /app/data/model.joblib
COPY ./src/movies.csv /app/data/movies.csv
COPY ./src/ratings.npz /app/data/ratings.npz

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
