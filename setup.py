"""
A setuptools based setup module for LCODE.
It uses pbr (http://docs.openstack.org/developer/pbr),
so most of the packaging metadata is moved to
setup.cfg and requirements.txt.
"""

from setuptools import setup


setup(
    setup_requires=[
        'cython>=0.21',
        'mako>=1.0',
        'numpy>=1.8.2',
        'pbr>=1.9',
        'setuptools>=20',
    ],
    pbr=True,
)
