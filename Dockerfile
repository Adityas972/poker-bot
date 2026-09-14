FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# CPU-only torch wheel -- the default PyPI build pulls in CUDA libraries
# that balloon the image by several GB and aren't usable on a free-tier
# host anyway.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# The pretrained policy network isn't checked into git (it's a binary
# blob that regenerates via `train_bot.py --big`, and training inside
# every Docker build would take ~25-30 min). Fetched instead from a
# GitHub Release asset -- see the README's "Deploying" section for how
# to publish a new one.
ARG MODEL_URL=https://github.com/Adityas972/poker-bot/releases/download/model-v1/policy_net.pt
RUN mkdir -p poker_bot/models && curl -fL "$MODEL_URL" -o poker_bot/models/policy_net.pt

ENV PORT=5050
EXPOSE 5050

# -w 1: SESSIONS lives in one process's memory (see poker_bot/web/app.py)
# -- more worker processes would fragment it, and a player's next
# request could land on a worker that's never heard of their hand.
# --threads gives real concurrency across players without that problem.
CMD ["sh", "-c", "gunicorn -w 1 --threads 4 -b 0.0.0.0:$PORT poker_bot.web.app:app"]
