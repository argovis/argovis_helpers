# test environment; versions match what Pyodide 314.0.8 supplies
# (https://pyodide.org/en/stable/usage/packages-in-pyodide.html),
# so tests here see the same stack a JupyterLite notebook will.
FROM python:3.14

RUN pip install pytest gsw \
    requests==2.33.1 \
    numpy==2.4.6 \
    scipy==1.18.0 \
    pandas==3.0.2 \
    xarray==2026.2.0 \
    shapely==2.1.2 \
    cftime==1.6.5
WORKDIR /app
COPY . .
