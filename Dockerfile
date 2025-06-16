FROM nvidia/cuda:12.0.1-runtime-ubuntu22.04

RUN apt-get update
RUN apt-get install -y python3 python3-pip git wget
RUN git clone https://github.com/ITMO-NSS-team/torch_DE_solver.git --branch rlpinn_new_arch  --single-branch
RUN cd torch_DE_solver && pip3 install .