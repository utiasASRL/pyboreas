import os
from setuptools import setup

from pathlib import Path

this_directory = Path(__file__).parent

with open(str(this_directory / 'README.md'), 'r') as f:
	long_description = f.read()

with open(str(this_directory / 'requirements.txt'), 'r') as f:
	requirements = f.read().splitlines()

setup(
	name='asrl-pyboreas',
	version='1.0.1',
	description='A toolkit for working with the Boreas dataset in Python',
	long_description=long_description,
	long_description_content_type='text/markdown',
	author='Keenan Burnett, Shichen Lu, Jingxing Qian, Yuchen Wu, David Yoon',
	author_email='keenan.burnett@robotics.utias.utoronto.ca',
	url='https://github.com/utiasASRL/boreas-devkit',
	license='BSD',
	packages=['pyboreas'],
	python_requires='>=3.8',
	install_requires=requirements
)
