FROM python:3.9

# Set the working directory in the container
WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 1313

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "1313", "--reload"]

# TO RUN:
# docker build -t testy . && docker container run -p 1313:1313 testy