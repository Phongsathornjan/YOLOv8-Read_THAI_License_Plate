FROM python:3.12.8-slim

WORKDIR /python-docker

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["waitress-serve", "--port=5000", "app:app"]