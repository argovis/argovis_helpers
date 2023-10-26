FROM argovis/argovis_helpers:test-base-231026

WORKDIR /app
COPY . .
CMD nosetests tests/*.py