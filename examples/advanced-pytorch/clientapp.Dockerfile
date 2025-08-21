FROM flwr/clientapp:1.19.0

WORKDIR /app
COPY pyproject.toml .
RUN sed -i 's/.*flwr\[simulation\].*//' pyproject.toml \
    && python -m pip install -U --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu .

ENTRYPOINT ["flwr-clientapp"]
