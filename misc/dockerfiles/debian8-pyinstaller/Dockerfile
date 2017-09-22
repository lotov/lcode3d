FROM python:3.5
# python:3.5 is based off jessie (debian 8) at the moment
MAINTAINER LCODE team "docker@lcode.info"
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
	libopenmpi-dev \
	openmpi-bin \
	upx-ucl \
	&& apt-get clean
RUN pip3 install pyinstaller
RUN pip3 install --only-binary=:all: \
	cython \
	h5py \
	nose \
	numpy \
	scipy
RUN pip3 install mako
RUN python3 -m pip download -d /tmp \
	hacks \
	matplotlib \
	mpi4py \
	&& rm /tmp/*.whl
