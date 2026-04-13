FROM gitea.little.local:30380/reeves/stt_server:torch

RUN pip3 install --no-cache-dir whisperx fastapi uvicorn python-multipart

WORKDIR /app
COPY app/ .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
