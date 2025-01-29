FROM python:3.9

RUN pip install requests pytest area numpy scipy shapely==1.8.0 geopandas
WORKDIR /app
COPY . .
