# Base Ubuntu Docker Image that we are building off of
# added open-cv for v1 modeling and cv2 tool kit for image resizing
# mpi4py installation problem solved by https://gist.github.com/lukoshkin/034dc718fdf2baff5ab216e487bbd831
FROM walkerlab/pytorch-jupyter:cuda-11.6.1-pytorch-1.12.0-torchvision-0.12.0-torchaudio-0.11.0-ubuntu-20.04
LABEL maintainer="diffusion_with_classifier"
RUN apt-get update \
    && apt-get install -y libopenmpi-dev 
RUN pip install blobfile mpi4py opencv-python
