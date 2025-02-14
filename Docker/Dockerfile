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
    libopenni2-0 \
    openni2-utils \
    zlib1g-dev \
    libzbar0 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --cache-dir /root/.cache/pip -r Requirements.txt
CMD ["python3" , "newalgo.py", "-u"]
