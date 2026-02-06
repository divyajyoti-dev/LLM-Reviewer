FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY data ./data

ENV PYTHONPATH=/app/src
RUN mkdir -p /app/outputs

CMD ["python", "-m", "reviewer_sim.run"]
