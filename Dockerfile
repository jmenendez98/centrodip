FROM python:3.12

RUN apt-get update && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --upgrade pip

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir .

CMD ["bash"]