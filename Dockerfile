FROM python:latest
WORKDIR /usr/app/src
COPY dnn.py ./
CMD [ "python", "./dnn.py"]