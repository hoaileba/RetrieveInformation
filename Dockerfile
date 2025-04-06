FROM python:3.10

WORKDIR /app

RUN pip install --upgrade pip setuptools wheel

RUN apt update && apt install -y curl build-essential && \
    curl https://sh.rustup.rs -sSf | sh -s -- -y && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install -U FlagEmbedding

COPY . .

RUN python -m pip install -e .