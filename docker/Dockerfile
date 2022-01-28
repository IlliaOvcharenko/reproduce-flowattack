# FROM nvidia/cuda:11.2.0-cudnn8-devel-ubuntu18.04
ARG PYTORCH="1.8.0"
ARG CUDA="11.1"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install MMCV
# RUN pip install mmcv-full==1.3.15 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html
RUN pip install mmcv-full==1.3.15 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html

# Install MMFlow
RUN conda clean --all
RUN git clone https://github.com/open-mmlab/mmflow.git /mmflow
WORKDIR /mmflow
ENV FORCE_CUDA="1"
RUN pip install -r requirements/build.txt
RUN pip install --no-cache-dir -e .

# # Create user
# ARG UID=1001
# ARG USER_NAME="dev"
# RUN useradd -u $UID -m $USER_NAME && \
#     echo "${USER_NAME}: " | chpasswd && adduser $USER_NAME sudo && \
#     echo "${USER_NAME} ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/$USER_NAME && \
#     chmod 0440 /etc/sudoers.d/$USER_NAME
# USER $USER_NAME
# WORKDIR /home/$USER_NAME


# # Install dotfiles
# RUN git clone https://github.com/IlliaOvcharenko/dotfiles
# RUN cd dotfiles && ./install && cd ../
# RUN echo dprompt >> /home/$USER_NAME/.bashrc


CMD ["/bin/bash"]

