# NVIDIA-PyTorch mit GB10/Blackwell-tauglichem CUDA-Stack
FROM nvcr.io/nvidia/pytorch:25.04-py3

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY convert.py app.py ./

# uvicorn auf Container-Port 80 (Compose mappt 8082:80)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
