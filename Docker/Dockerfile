FROM python:slim-buster
COPY . /app
WORKDIR /app
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libxrender-dev \
    libx11-dev \
    libatlas-base-dev \
    libopenni2-dev \
    zlib1g-dev \
    libzbar0 \
    && rm -rf /var/lib/apt/lists/*
pip install --cache-dir /root/.cache/pip -r requirements.txt
CMD ["python3" , "newalgo.py", "-u"]
