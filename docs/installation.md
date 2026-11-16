# Installation & Development Environment

This project was developed in a containerized environment using **WSL2**, **Docker Desktop**, and **Visual Studio Code Dev Containers**. This keeps the development environment consistent and makes it easier to manage the required Python packages and GPU dependencies.

## Prerequisites

Before getting started, make sure you have the following installed:

- Windows with WSL2 enabled
- A Linux distribution installed through WSL2
- Docker Desktop with WSL integration enabled
- Visual Studio Code
- The Visual Studio Code **Dev Containers** extension
- NVIDIA GPU drivers and a working CUDA setup

## Clone the Repository

Open a terminal in your WSL2 environment and clone the repository:

```bash
git clone https://github.com/rick-degraaf/fruit-vision.git
cd fruit-vision
```

## Open the Project in a Dev Container

1. Open the project folder in Visual Studio Code.
2. Open the Command Palette with `Ctrl + Shift + P`.
3. Select **Dev Containers: Rebuild and Reopen in Container**.
4. Wait for the container to build and start.

Visual Studio Code will reopen the project inside the development container.

## GPU Support

The project uses PyTorch and Ultralytics YOLOv8 for model training and inference. A CUDA-compatible environment is required to use the NVIDIA GPU.

Once the container is running, you can check whether PyTorch detects CUDA:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

A result of `True` indicates that CUDA is available to PyTorch.

## Development Environment

The project uses the following main technologies:

- Python
- PyTorch
- Ultralytics YOLOv8
- NumPy
- Docker
- Roboflow
- Visual Studio Code

For dataset preparation, training, inference, and evaluation, refer to the corresponding documentation files in this folder.