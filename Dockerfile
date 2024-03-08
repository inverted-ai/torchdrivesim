FROM nvidia/cudagl:11.4.2-devel-ubuntu20.04 as torchdrivesim-base
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

# Install pytorch, pytorch3d and lanelet2
RUN pip install --no-cache-dir \
    torch==2.1.2+cu118 torchvision==0.16.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html \
    pytorch3d==0.7.5 -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt210/download.html \
    lanelet2

# Install Lanelet2 dependencies
RUN apt update \
  && apt install -y libboost-all-dev libeigen3-dev libgeographic-dev libpugixml-dev libboost-python-dev


WORKDIR /opt


FROM torchdrivesim-base as torchdrivesim-tests

COPY tests /opt/tests
COPY requirements/dev.txt /opt/requirements.txt
RUN pip install --no-cache-dir -r /opt/requirements.txt
COPY . /opt/torchdrivesim
RUN cd torchdrivesim && pip install .
COPY pytest.ini /opt/pytest.ini
CMD ["pytest", "-s", "-m", "not depends_on_nvdiffrast", "tests"]

FROM torchdrivesim-base as torchdrivesim

# Install torchdrivesim
COPY . /opt/torchdrivesim
WORKDIR /opt/torchdrivesim
RUN pip install build && python -m build --sdist --wheel --outdir dist && pip install dist/$(ls dist/ | grep " *.whl")[dev,tests] && rm -R ../torchdrivesim
WORKDIR /
