FROM argovis/argovis_helpers:test-base-230412

WORKDIR /app
COPY . .
CMD nosetests tests/*.py