FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04

# install python3-pip
RUN apt update && apt install python3-pip git vim sudo curl wget apt-transport-https ca-certificates gnupg libgl1 -y 

RUN pip install setuptools
# install dependencies via pip
# Only install jax/jaxlib to version 0.4.11 for st
RUN pip3 install pandas numpy scipy six wheel jax[cuda]==0.4.11 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html jaxlib==0.4.11 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

RUN pip3 install distrax brax chex flax optax gym  notebook matplotlib tqdm gymnax jupyter ipython

RUN pip3 install wandb

ARG UID
RUN useradd -u $UID --create-home duser && \
    echo "duser:duser" | chpasswd && \
    adduser duser sudo
USER duser
WORKDIR /home/duser/
