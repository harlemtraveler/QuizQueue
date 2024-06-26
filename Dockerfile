FROM --platform=linux/amd64 python:3.11.8
#FROM python:3.11.8

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade -r requirements.txt

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

COPY . .

ARG OPENAI_API_KEY
ARG HUGGINGFACEHUB_API_TOKEN

ENV OPENAI_API_KEY=$OPENAI_API_KEY
ENV HUGGINGFACEHUB_API_TOKEN=$HUGGINGFACEHUB_API_TOKEN

EXPOSE 5000

CMD ["python", "app.py"]

