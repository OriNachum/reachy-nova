FROM python:3.12-slim

WORKDIR /app

COPY docker/nervous-system/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY docker/nervous-system/nervous_system.py .
COPY docker/nervous-system/heartbeat.py .
COPY config/nervous-system/rules.yaml /app/config/rules.yaml

CMD ["python", "nervous_system.py"]
