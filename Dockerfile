FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./
COPY src/ ./src/
COPY config/ ./config/
COPY models/ ./models/
COPY run_setup.py run_train.py run_inference.py entrypoint.sh ./

RUN pip install --no-cache-dir -e . && chmod +x entrypoint.sh

ENV RUNNING_IN_DOCKER=1

EXPOSE 8000

ENTRYPOINT ["./entrypoint.sh"]
CMD ["api"]
