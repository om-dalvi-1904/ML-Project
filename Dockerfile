FROM python:3.13-alpine
WORKDIR /app
COPY . /app
RUN apt update -y && apt install awscli -y

RUN pip install -r requirements.txt
CMD [ "python3", "app.py"]