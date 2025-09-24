# Use the same base image version as the clams-python python library version
FROM ghcr.io/clamsproject/clams-python-ffmpeg-hf:1.3.3
# See https://github.com/orgs/clamsproject/packages?tab=packages&q=clams-python for more base images
# IF you want to automatically publish this image to the clamsproject organization, 
# 1. you should have generated this template without --no-github-actions flag
# 1. to add arm64 support, change relevant line in .github/workflows/container.yml 
#     * NOTE that a lots of software doesn't install/compile or run on arm64 architecture out of the box 
#     * make sure you locally test the compatibility of all software dependencies before using arm64 support 
# 1. use a git tag to trigger the github action. You need to use git tag to properly set app version anyway

################################################################################
# DO NOT EDIT THIS SECTION
ARG CLAMS_APP_VERSION
ENV CLAMS_APP_VERSION ${CLAMS_APP_VERSION}
################################################################################

################################################################################
# clams-python base images are based on debian distro
# install more system packages as needed using the apt manager
################################################################################

# Install build tools required for bitsandbytes and flash-attn compilation
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install CUDA development tools for flash-attn compilation
# Note: This is a large download and will significantly increase image size
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb && \
    apt-get update && \
    apt-get install -y cuda-toolkit-12-1 && \
    rm -rf /var/lib/apt/lists/* && \
    rm cuda-keyring_1.0-1_all.deb

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda-12.1
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

################################################################################
# main app installation
COPY ./ /app
WORKDIR /app
RUN pip3 install -r requirements.txt

# default command to run the CLAMS app in a production server 
CMD ["python3", "app.py", "--production"]
################################################################################
