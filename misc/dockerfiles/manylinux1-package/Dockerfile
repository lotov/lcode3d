FROM quay.io/pypa/manylinux1_x86_64
MAINTAINER LCODE team "docker@lcode.info"
RUN /opt/python/cp35-cp35m/bin/easy_install-3.5 pip
RUN /opt/python/cp35-cp35m/bin/python3.5 -m pip install --upgrade setuptools
RUN /opt/python/cp35-cp35m/bin/python3.5 -m pip install --only-binary=:all: \
	cython \
	h5py \
	nose \
	numpy==1.9.3
RUN /opt/python/cp35-cp35m/bin/python3.5 -m pip install mako

RUN /opt/python/cp36-cp36m/bin/easy_install-3.6 pip
RUN /opt/python/cp36-cp36m/bin/python3.6 -m pip install --upgrade setuptools
RUN /opt/python/cp36-cp36m/bin/python3.6 -m pip install --only-binary=:all: \
	cython \
	h5py \
	nose \
	numpy
RUN /opt/python/cp36-cp36m/bin/python3.6 -m pip install mako
