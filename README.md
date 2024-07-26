# Tiny-DeepSpeed

Welcome to Tiny-DeepSpeed, a minimalistic implementation of the DeepSpeed library. This project is designed to provide a simple, easy-to-understand codebase that helps learners and developers understand the core functionalities of DeepSpeed, a powerful library for accelerating deep learning models.

## Project Overview

Tiny-DeepSpeed simplifies the complex mechanisms of the original DeepSpeed library, focusing on its key components to offer a straightforward starting point for educational purposes and experimentation. This project is perfect for those new to model optimization and distributed training.

## Features

- **Simplified Codebase**: Just the essentials to get you started with DeepSpeed.
- **Educational Tool**: Ideal for teaching, learning, and experimentation.
- **Scalability**: Demonstrates basic principles that can scale with more complex implementations.

## Getting Started

### Prerequisites

Before you begin, ensure you have the following installed:
- Python 3.11
- PyTorch (CUDA) 2.3.1

## Installation
Clone this repository to your local machine:
```bash
git clone https://github.com/liangyuwang/Tiny-DeepSpeed.git
cd Tiny-DeepSpeed
```

## Running the Demo
To run the Tiny-DeepSpeed demo, use the following command:
```bash
# Single Device
python test/single_device/train.py

# DDP mode
torchrun --nproc_per_node $num_device$ --nnodes 1 test/ddp/train.py

# Zero1 mode
torchrun --nproc_per_node $num_device$ --nnodes 1 test/zero1/train.py
```
This will initiate a simple training loop using the Tiny-DeepSpeed framework.

## TODO:

- [X] Single Device
- [X] DDP
- [X] Zero1
- [ ] Zero2
- [ ] Zero3
- [ ] AMP support
- [ ] Compute-communication overlap
- [ ] Multi nodes
