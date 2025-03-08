FROM python:slim-buster

COPY . /app
WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    libusb-1.0-0-dev \
    libturbojpeg0-dev \
    libglfw3-dev \
    git \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libxrender-dev \
    libx11-dev \
    libatlas-base-dev \
    libxi6 \
    libxmu6 \
    && rm -rf /var/lib/apt/lists/*

# Install libfreenect2
RUN cd ~ && git clone https://github.com/OpenKinect/libfreenect2.git \
    && cd libfreenect2 \
    && mkdir build && cd build \
    && cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local \
    && make -j$(nproc) \
    && make install

# Set environment variables for libfreenect2
ENV CPLUS_INCLUDE_PATH=/usr/local/include
ENV LIBRARY_PATH=/usr/local/lib
ENV LD_LIBRARY_PATH=/usr/local/lib


# Install pylibfreenect2
RUN pip install numpy opencv-python cython
RUN cd ~ && git clone https://github.com/r9y9/pylibfreenect2.git \
    && cd pylibfreenect2 \
    && pip install .

CMD ["python3", "kinectv2.py", "-u"]

