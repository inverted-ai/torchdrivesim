[pypi-badge]: https://badge.fury.io/py/torchdrivesim.svg
[pypi-link]: https://pypi.org/project/torchdrivesim/  


[![CI](https://github.com/inverted-ai/torchdrivesim/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/inverted-ai/torchdrivesim/actions/workflows/CI.yml)
[![PyPI][pypi-badge]][pypi-link]
# TorchDriveSim

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

TorchDriveSim uses Lanelet2 as its map format, but it runs without Lanelet2 on any of the pre-defined
maps available as meshes in this repo, although it won't be able to detect wrong way infractions and use certain
heuristic behavioral models. To use those features, and to use TorchDriveSim with your own maps, you'll need to install
Lanelet2 with its Python bindings. You can either use the official distribution or the fork hosted by Inverted AI,
which allows for installation without ROS.

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

TorchDriveSim supports two rendering backends, namely pytorch3d and nvdiffrast, both producing results that look the
same to the human eye. Pytorch3d is the default one and a required dependency, since it's easier to install. Nvdiffrast
is supported and can sometimes be substantially faster, but it needs to be installed separately, and it's subject
to more restrictive license conditions. We also provide a dummy rendering backend that returns an empty image,
mostly for debugging and benchmarking purposes.

## Installation

Before running the usual `pip install torchdrivesim` command, correct `torch` and `pytorch3d` versions need to be 
installed by the user. `torchdrivesim` will assume the user have installed
the correct version already, otherwise the default versions of `torch` and `pytorch3d` will be installed which usually
will cause incompatibility errors.   

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
