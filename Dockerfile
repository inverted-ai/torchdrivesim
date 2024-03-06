FROM nvidia/cuda:11.7.1-devel-ubuntu20.04 as rl-env-cuda-117
# This file is for building the production api server only.

# Install general utilities, Python, related tooling, C compilers, build systems, and Lanelet2 dependencies in one go
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    tzdata software-properties-common locales-all git curl sed nano \
    libffi-dev  \
    build-essential llvm libxml2-dev cmake autoconf ninja-build \
    clang-8 lld-8 g++-7 libpng-dev libtiff5-dev libjpeg-dev \
    libtool libxml2-dev libxerces-c-dev libboost-all-dev libeigen3-dev \
    libgeographic-dev libpugixml-dev libboost-python-dev --no-install-recommends \
 && add-apt-repository ppa:deadsnakes/ppa \
 && apt-get update && apt-get install -y python3.10 python3.10-venv python3.10-dev wget\
 && wget https://bootstrap.pypa.io/get-pip.py && python3.10 get-pip.py \
 && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
 && pip install --no-cache-dir setuptools distro wheel \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

ENV DEBIAN_FRONTEND=

RUN apt install -y nvidia-modprobe

RUN SNIPPET="export PROMPT_COMMAND='history -a' && export HISTFILE=/root/.bash_history" \
    && echo "$SNIPPET" >> "/root/.bashrc"

# Install pytorch and pytorch3d
RUN pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu117_pyt1131/download.html

# Set up C compilers and build systems
RUN apt update && apt install -y build-essential cmake autoconf ninja-build \
  clang-8 lld-8 g++-7 \
  libpng-dev libtiff5-dev libjpeg-dev libtool libxml2-dev libxerces-c-dev libgl1

# Install Lanelet2
RUN apt update \
  && apt install -y libboost-all-dev libeigen3-dev libgeographic-dev libpugixml-dev libboost-python-dev

RUN pip install opencv-python
RUN pip install gymnasium[classic-control] stable-baselines3
RUN pip install wandb tensorboard moviepy
RUN pip install jupyter notebook
RUN pip install omegaconf scipy shapely
RUN pip install invertedai


# WORKDIR /opt
#
#
# FROM rl-env-base as torchdrivesim-tests
#
# COPY tests /opt/tests
# COPY requirements/dev.txt /opt/requirements.txt
# RUN pip install -r /opt/requirements.txt
# COPY pytest.ini /opt/pytest.ini
# CMD ["pytest", "-s", "-m", "not depends_on_cuda", "tests"]
#
# FROM torchdrivesim-base as torchdrivesim
#

# Install torchdrivesim
# COPY . /opt/torchdrivesim
# WORKDIR /opt/torchdrivesim
# RUN pip install build && python -m build --sdist --wheel --outdir dist && pip install dist/$(ls dist/ | grep " *.whl")[dev,tests]
# WORKDIR /
# RUN pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels//py38_cu116_pyt1131download.html

# Set up C compilers and build systems
RUN apt update && apt install -y build-essential cmake autoconf ninja-build \
  clang-8 lld-8 g++-7 \
  libpng-dev libtiff5-dev libjpeg-dev libtool libxml2-dev libxerces-c-dev libgl1

# Install Lanelet2
RUN apt update \
  && apt install -y libboost-all-dev libeigen3-dev libgeographic-dev libpugixml-dev libboost-python-dev
COPY resources/dependencies/lanelet2/lanelet2-0.1.0-cp38-cp38-linux_x86_64.whl /opt/lanelet2-0.1.0-cp38-cp38-linux_x86_64.whl
RUN pip install /opt/lanelet2-0.1.0-cp38-cp38-linux_x86_64.whl

RUN pip install opencv-python
RUN pip install gymnasium[classic-control] stable-baselines3
RUN pip install wandb tensorboard moviepy
RUN pip install jupyter notebook
RUN pip install omegaconf scipy shapely
RUN pip install invertedai


# WORKDIR /opt
#
#
# FROM rl-env-base as torchdrivesim-tests
#
# COPY tests /opt/tests
# COPY requirements/dev.txt /opt/requirements.txt
# RUN pip install -r /opt/requirements.txt
# COPY pytest.ini /opt/pytest.ini
# CMD ["pytest", "-s", "-m", "not depends_on_cuda", "tests"]
#
# FROM torchdrivesim-base as torchdrivesim
#
# Install torchdrivesim
# COPY . /opt/torchdrivesim
# WORKDIR /opt/torchdrivesim
# RUN pip install build && python -m build --sdist --wheel --outdir dist && pip install dist/$(ls dist/ | grep " *.whl")[dev,tests]
