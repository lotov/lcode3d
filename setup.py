"""
A setuptools based setup module for LCODE.
It uses pbr (http://docs.openstack.org/developer/pbr),
so most of the packaging metadata is moved to
setup.cfg and requirements.txt.
"""

from setuptools import setup


setup(
    setup_requires=[
        'pbr>=1.9',
    ],
    pbr=True,
)
