FROM python:3.9

RUN pip install requests pytest area numpy scipy
WORKDIR /app
COPY . .
