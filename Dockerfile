FROM argovis/argovis_helpers:test-base-220907

WORKDIR /app
COPY . .
CMD nosetests tests/*.py