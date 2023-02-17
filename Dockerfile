FROM nvidia/cudagl:11.2.2-devel-ubuntu20.04 as torchdrivesim-base
# This file is for building the production api server only.

# Install general utilities
RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y tzdata software-properties-common && add-apt-repository universe && apt update && apt install -y locales-all  git curl sed nano

# Install Python and related tooling
RUN apt install -y libffi-dev python python-dev python3-dev python3-pip python3-venv \
  && update-alternatives --install /usr/bin/python python /usr/bin/python3 1 \
  && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1 \
  && pip install setuptools distro wheel \
  && apt install -y build-essential llvm libxml2-dev

# Install pytorch and pytorch3d
RUN pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install pytorch3d==0.7.2 -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1110/download.html

# Set up C compilers and build systems
RUN apt update && apt install -y build-essential cmake autoconf ninja-build \
  clang-8 lld-8 g++-7 \
  libpng-dev libtiff5-dev libjpeg-dev libtool libxml2-dev libxerces-c-dev

# Install Lanelet2
RUN apt update \
  && apt install -y libboost-all-dev libeigen3-dev libgeographic-dev libpugixml-dev libboost-python-dev
COPY resources/dependencies/lanelet2/lanelet2-0.1.0-cp38-cp38-linux_x86_64.whl /opt/lanelet2-0.1.0-cp38-cp38-linux_x86_64.whl
RUN pip install /opt/lanelet2-0.1.0-cp38-cp38-linux_x86_64.whl


WORKDIR /opt


FROM torchdrivesim-base as torchdrivesim-tests

COPY tests /opt/tests
COPY requirements/dev.txt /opt/requirements.txt
RUN pip install -r /opt/requirements.txt
COPY pytest.ini /opt/pytest.ini
CMD ["pytest", "-s", "-m", "not depends_on_cuda", "tests"]

FROM torchdrivesim-base as torchdrivesim

# Install torchdrivesim
COPY . /opt/torchdrivesim
WORKDIR /opt/torchdrivesim
RUN pip install build && python -m build --sdist --wheel --outdir dist && pip install dist/$(ls dist/ | grep " *.whl")[dev,tests] && rm -R ../torchdrivesim
WORKDIR /
