FROM python:3.9-slim-buster

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 1313

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "1313", "--reload"]
