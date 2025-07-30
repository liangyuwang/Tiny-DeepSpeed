# Tiny-DeepSpeed

Welcome to Tiny-DeepSpeed, a minimalistic re-implementation of the DeepSpeed library. This project is designed to provide a simple, easy-to-understand codebase that helps learners and developers understand the core functionalities of [DeepSpeed](https://github.com/microsoft/DeepSpeed), a powerful library for accelerating deep learning models.

Share us a ⭐ if this github repo does help.

The table below shows the training GPU memory (GB) of GPT2 under different parallelism strategies for performance comparison.
|        Methods        |   1 GPU   |   DDP - 2 GPU   |   Zero1 - 2 GPU   |   Zero2 - 2 GPU   |   Zero3 - 2 GPU   |
| :-----------------------: | :------: | :------: | :------: | :------: | :------: |
| **GPT2-small** | `4.65` | `4.75` | `4.08` | `3.79` | `3.69` |
| **GPT2-medium** | `10.12` | `10.23` | `8.65` | `8.25` | `7.73` |
| **GPT2-large** | `17.35` | `17.46` | `14.08` | `12.89` | `11.01` |

If you encounter any question, please feel free to contact us. You can create an issue or just send an email to me at [liangyu.wang@kaust.edu.sa](liangyu.wang@kaust.edu.sa).

## Getting Started

### Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.11
- PyTorch (CUDA) 2.3.1
- triton 2.3.1

## Installation

Clone this repository to your local machine:

```bash
git clone https://github.com/liangyuwang/Tiny-DeepSpeed.git
cd Tiny-DeepSpeed
```

## Running the Demo

To run the Tiny-DeepSpeed demo, use the following command (set "num_device" to your number of devices):

```bash
# Single Device
python example/single_device/train.py

# DDP mode
torchrun --nproc_per_node num_device --nnodes 1 example/ddp/train.py

# Zero1 mode
torchrun --nproc_per_node num_device --nnodes 1 example/zero1/train.py

# Zero2 mode
torchrun --nproc_per_node num_device --nnodes 1 example/zero2/train.py

# Zero3 mode
torchrun --nproc_per_node num_device --nnodes 1 example/zero3/train.py
```

This will initiate a simple training loop using the Tiny-DeepSpeed framework.

Feel free to try our demo online on [Kaggle Notebook](https://www.kaggle.com/code/wlykaggle/tiny-deepspeed-example)

## Features

- **Simplified Codebase**: Stripped down to the essential components to facilitate learning and experimentation with DeepSpeed.
- **Meta Device Model Initialization**: Loads model parameters on a meta device, avoiding actual parameter initialization and reducing initial memory usage.
- **Parameter Distribution via Cache Rank Map**: Implements a cache rank map table to distribute model parameters across different ranks. Each parameter is assigned a rank ID based on the number of participants, allowing for efficient and targeted initialization.
- **Scalability and Flexibility**: Demonstrates basic principles of distributed training and parameter management that can be scaled up for more complex implementations.
- **Educational Tool**: Serves as a practical guide for those new to model optimization and distributed computing in machine learning.

## TODO:

- [X] Single Device
- [X] DDP
- [X] Zero1
- [X] Zero2
- [X] Zero3
- [ ] AMP support
- [X] Compute-communication overlap
- [X] Meta initialization
- [ ] Multi nodes
- [ ] Communication Bucket
