#-------------------------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See https://go.microsoft.com/fwlink/?linkid=2090316 for license information.
#-------------------------------------------------------------------------------------------------------------

FROM tensorflow/tensorflow:2.3.0-gpu

# Avoid warnings by switching to noninteractive
ENV DEBIAN_FRONTEND=noninteractive

# This Dockerfile adds a non-root user with sudo access. Use the "remoteUser"
# property in devcontainer.json to use it. On Linux, the container user's GID/UIDs
# will be updated to match your local UID/GID (when using the dockerFile property).
# See https://aka.ms/vscode-remote/containers/non-root-user for details.
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

ENV PIP_TARGET=/usr/local/share/pip-global
ENV PYTHONPATH=${PYTHONPATH}:${PIP_TARGET}
ENV PATH=${PATH}:${PIP_TARGET}/bin

# Uncomment the following COPY line and the corresponding lines in the `RUN` command if you wish to
# include your requirements in the image itself. It is suggested that you only do this if your
# requirements rarely (if ever) change.
# COPY requirements.txt /tmp/pip-tmp/

# Configure apt and install packages
RUN apt-get update -y \
    &&  apt-get install -y  sudo \
    &&  apt-get install -y  git \
    &&  apt-get install -y  libopenexr-dev \
    && apt-get -y install --no-install-recommends apt-utils dialog 2>&1 \
    #
    # Verify git, process tools, lsb-release (common in install instructions for CLIs) installed
    && apt-get -y install git iproute2 procps lsb-release \
    #
    # Install pylint
    && pip --disable-pip-version-check --no-cache-dir install pylint \
    #
    # Update Python environment based on requirements.txt
    # && pip --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements.txt \
    # && rm -rf /tmp/pip-tmp \
    #
    # Create a non-root user to use if preferred - see https://aka.ms/vscode-remote/containers/non-root-user.
    && groupadd --gid $USER_GID $USERNAME \
    && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
    # [Optional] Add sudo support for the non-root user
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME\
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    #
    # Create alternate global install location that both uses have rights to access
    && mkdir -p /usr/local/share/pip-global \
    && chown ${USERNAME}:root /usr/local/share/pip-global \
    #
    # Clean up
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update -y \
 && pip3 install scikit-image \
 && pip3 install autopep8 \
 && pip3 install pyyaml \
 && pip3 install --upgrade tensorflow-graphics-gpu \
 && pip3 install --upgrade OpenEXR \
 && pip3 install tensorflow-addons \
 && echo 'root:Docker!' | chpasswd

RUN pip install colour
RUN pip install -U matplotlib

#RUN apt-get update -y \
#    && pip3 install opencv-python-headless

VOLUME data
VOLUME database
VOLUME project

# Switch back to dialog for any ad-hoc use of apt-get
ENV DEBIAN_FRONTEND=dialog


