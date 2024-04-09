[pypi-badge]: https://badge.fury.io/py/torchdrivesim.svg
[pypi-link]: https://pypi.org/project/torchdrivesim/  
[python-badge]: https://img.shields.io/pypi/pyversions/torchdrivesim.svg?color=%2334D058
[![CI](https://github.com/inverted-ai/torchdrivesim/actions/workflows/CI_cpu.yml/badge.svg?branch=master)](https://github.com/inverted-ai/torchdrivesim/actions/workflows/CI_cpu.yml)
[![PyPI][pypi-badge]][pypi-link]
[![python-badge]][pypi-link]
[![Documentation Status](https://readthedocs.org/projects/torchdrivesim/badge/?version=latest)](https://docs.torchdrivesim.org/en/latest/)

# TorchDriveSim
<!-- start Features-->
TorchDriveSim is a lightweight 2D driving simulator, built entirely in [PyTorch](https://pytorch.org/), primarily intended as a training
environment for developing autonomous driving algorithms. Its main features are:
1. Fully differentiable execution producing a single computation graph, including state transition (kinematic models) and observation (differentiable rendering) models.
2. First class support for batch processing, down to the lowest level.
3. Support for heterogeneous agent types (vehicles, pedestrians, cyclists, etc.), each with its own kinematic model.
4. Extensible and customizable implementation of kinematic models (unconstrained, unicycle, bicycle, etc.), rendering modes, and rendering backends.
5. Support for extensible traffic control types, including traffic lights.
6. Differentiable implementations of various infraction metrics (collisions, off-road, wrong way).
7. Modular collection of wrappers modifying the simulator's behavior.
8. Ability to ingest any map in [Lanelet2](https://github.com/fzi-forschungszentrum-informatik/Lanelet2) format out of the box.
9. Integration with [IAI API](https://docs.inverted.ai/en/latest/) for initializing agent states and providing realistic behaviors.
<!-- end Features-->
### ** Warning **
The PyPI version of torchdrivesim (which is what you get when you run `pip install torchdrivesim`) comes equipped
only with the slower OpenCV renderer. Faster renderers are available, but require either 
[`pytorch3d`](https://github.com/facebookresearch/pytorch3d/) or 
[`nvdiffrast`](https://nvlabs.github.io/nvdiffrast/)
to be installed. Their CUDA dependencies can be tricky to satisfy, so we provide a suitable Dockerfile.

<!-- start readme-->
## Simulator Architecture
The simulated world consists of the following elements:
1. Static background, by default including road (drivable surface) and lane markings, represented as a triangular mesh.
2. Control elements, represented by rectangles with internal state. The simulator does not enforce their semantics.
3. A collection of agents grouped into arbitrary types. All agents are rigid rectangles.
4. Per agent type kinematic models, defining the agents' action space and how actions translate into motion.
5. A configurable renderer, displaying the world form bird's eye view (birdview), using a customizable color map.

Each agent is described by its static attributes (length, width, and others as needed by kinematic models),
dynamic state (x, y, orientation, speed), and a flag (present mask) indicating whether a given agent is currently alive.
At each time step, the agents perform actions, which in turn determine
their next state. The simulator allows the agents to overlap, which is identified as a collision but not prohibited,
and the states of different agents do not influence each other except through the agents' actions. The simulator can
operate either in homogeneous mode (all agents are the same type and their states and actions are tensors), or in
heterogeneous mode (there are multiple agent types and their states and actions are dictionaries mapping agent
types to tensors). To support both modes, most operations are applied as functors, which lift a function acting on
a single agent type into a function acting on all agent types. However, this behavior should be transparent to users
who do not modify the simulator code.

The base simulator requires actions for all agents and does not modify their presence masks. For convenience, we provide
various wrappers modifying the simulator's behavior, such as by controlling a subset of agents (by replay or pre-defined
ontrollers), removing agents that exit the designated area, monitoring infractions, recording video, and many others.
Unless specified otherwise, the wrappers can be combined in different orders to reach desired effects.

## Behavioral Models

The hardest driving scenarios are those that require interactions with other road users. When building simulated
environments it is crucial to ensure that the other agents behave realistically, but achieving that is not easy and
TorchDriveSim is flexible in terms of how those other agents are controlled. We provide a simple heuristic that achieves
minimally sensible driving, noting that in most cases it will be unsatisfactory and requiring additional extensions.
We also facilitate log replay and demonstrate how to use recorded actions from the INTERACTION dataset. However,
such replay is not reactive, often resulting in spurious collisions.

For maximum realism and reactivity, we recommend using our (Inverted AI) API for generating realistic behaviors,
which is integrated with TorchDriveSim. This is a paid offering that requires an API key, which you can obtain by
contacting us. For academics, we may be able to offer free API keys.

## Maps and Map Formats

Several CARLA maps (`carla_Town01`, `carla_Town02`, `carla_Town03`, `carla_Town04`, `carla_Town06`, `carla_Town07`, `carla_Town10HD`)
are included in `torchdrivesim` itself and can be loaded
by name. To include other maps, place the files in the format described below somewhere in a folder referenced by the
`TDS_RESOURCE_PATH` environment variable. Generally, a map is defined by a folder with the following structure:
```
MAPNAME/
  metadata.json  # custom metadata format
  MAPNAME.osm  # road network in Lanelet2 format
  MAPNAME_mesh.json  # custom road mesh format, can be derived from the .osm file
  MAPNAME_stoplines.json  # custom format specifying traffic lights, stop signs, and yield signs, if needed
```

See the bundled maps in `torchdrivesim/resources/maps` for examples. There is currently no tooling available
for creating TorchDriveSim-compatible maps, but you can try the experimental OpenDRIVE
[converter](https://github.com/inverted-ai/map-converter).

## Scenario Definition

With maps and behavioral models available, the final hurdle is to define a suite of driving scenarios that can be
used for testing and evaluation. TorchDriveSim provides helpers for initializing the simulation state, including
by calling Inverted AI API, instantiating from a log (when available), and using some simple heuristics. It also
provides functions for identifying driving infractions, specifically collisions, going off-road, and driving wrong way.
However, it does not specify goals or rewards, leaving that to the user. Over time, we are planning to release
various scenario suites that can serve as benchmarks.

## Kinematic models

The primary kinematic model for vehicles in TorchDriveSim is the bicycle model, where the action consists of steering
and acceleration. It requires additionally specifying the rear axis offset to control the vehicle turn radius, but
it does not use the front axis offset, since that can not be fit by observing the vehicle movement from the outside,
effectively assuming the front axis is in the middle of the vehicle. Other kinematic models available are the
unconstrained model, where the action is the state delta between subsequent time steps, and the teleporting model,
where the action directly specifies the next state. We also provide different variations of those models, and it is
straightforward to implement a custom one.

## Differentiable rendering

TorchDriveSim supports three rendering backends, using cv2, pytorch3d, and nvdiffrast, respectively. The images produced
by all three look very similar and differ only in the details of how different triangles are pixelated.
The cv2 backend is the easiest to install and it is included as a required dependency. Pytorch3d and nvdiffrast
need to be installed separately, as per the instructions below. We also provide a dummy rendering backend that
returns an empty image, mostly for debugging and benchmarking purposes.

## Installation

Running `pip install torchdrivesim` only provides access to the basic OpenCV renderer. To be able to use the faster
pytorch3d renderer, make sure to first install the correct versions of `torch` and `pytorch3d` using the instructions
below. You can also install [`nvdiffrast`](https://nvlabs.github.io/nvdiffrast/#installation), which can be even faster,
but it is also subject to more restrictive license conditions.
Generally, the more images you render in parallel (either by having more ego agents or larger batches), the more
of a gain you will get from the faster renderers.

To install the correct `torch` using `pip`, go visit the
[prebuilt whls page](https://download.pytorch.org/whl/torch_stable.html) to select the right `.whl` file based on the
Python version, cuda availability and the operating system. For example, to install version `1.11.0` for `python3.8`
with `cuda` on Linux, run  
 `pip install https://download.pytorch.org/whl/cu113/torch-1.10.2%2Bcu113-cp38-cp38-linux_x86_64.whl`  
or  
`pip install torch==1.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html`  

To install the correct `pytorch3d`, the user need to find the correct prebuilt wheel for the installed `torch` version,
more details can be found at the
[official installation page](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md). For example, to 
install version `0.7.2` for `python3.8` with `pip`, run  
`pip install https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1110/pytorch3d-0.7.2-cp38-cp38-linux_x86_64.whl`  
or  
`pip install pytorch3d==0.7.2 -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1110/download.html`  

Here are the summarized example steps for python3.8 and cuda 11.3:
```bazaar
1. pip install torch==1.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
2. pip install pytorch3d==0.7.2 -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1110/download.html
3. pip install torchdrivesim
```
## Docker

In order to use TorchDriveSim smoothly without worrying about installing dependencies, we provide a 
[Dockerfile](Dockerfile) that works either with or without gpu. In order to build the docker image,
run `docker build --target torchdrivesim -t torchdrivesim:latest . `. To run the container with GPU access,
run `docker run --runtime=nvidia -it torchdrivesim:latest /bin/bash`. To run the container without GPU access,
run `docker run -it torchdrivesim:latest /bin/bash`. For more information regarding setting up GPU runtime with Docker,
follow [this official link](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
for the installation process.

## Citations

If you use TorchDriveSim in your research, please cite the following [paper](https://arxiv.org/abs/2104.11212),
for which an early version of TorchDriveSim  was initially developed.

```bibtex
@INPROCEEDINGS{itra2021,
  author={\'Scibior, Adam and Lioutas, Vasileios and Reda, Daniele and Bateni, Peyman and Wood, Frank},
  booktitle={2021 IEEE International Intelligent Transportation Systems Conference (ITSC)}, 
  title={Imagining The Road Ahead: Multi-Agent Trajectory Prediction via Differentiable Simulation}, 
  year={2021},
  pages={720-725},
  doi={10.1109/ITSC48978.2021.9565113}}
```
<!-- end readme-->
