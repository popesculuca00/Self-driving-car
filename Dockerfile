# FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04
FROM nvidia/cuda:12.3.0-base-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3.8
ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

RUN apt-get update && apt-get install -y \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1
RUN update-alternatives --set python3 /usr/bin/python${PYTHON_VERSION}

COPY simulation simulation
COPY static static
COPY templates templates
COPY utils utils
COPY main.py main.py
COPY requirements.txt requirements.txt

RUN python3.8 -m pip install --upgrade pip
RUN pip3 install -r requirements.txt
RUN pip3 install torch==2.3.1 torchvision==0.18.1 --extra-index-url https://download.pytorch.org/whl/cu123

EXPOSE 2000-2002 5000

CMD ["python3.8", "main.py"]