FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY nifty_scalper.py server.py ./
COPY docs/index.html ./docs/index.html
ENV DATA_DIR=/data PORT=8000 TZ=Asia/Kolkata
VOLUME /data
EXPOSE 8000
CMD ["python", "server.py"]
