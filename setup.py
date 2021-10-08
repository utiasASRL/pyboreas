import os
from setuptools import setup

with open('README.md', 'r') as f:
	long_description = f.read()

with open('requirements.txt', 'r') as f:
	requirements = f.read().splitlines()

setup(
	name='boreas-devkit',
	version='1.0.0',
	description='A toolkit for working with the Boreas dataset in Python',
	author='Keenan Burnett, Shichen Lu, Jingxing Qian, Yuchen Wu, David Yoon',
	author_email='keenan.burnett@robotics.utias.utoronto.ca',
	url='https://github.com/utiasASRL/boreas-devkit',
	license='BSD',
	packages=['pyboreas'],
	python_requires='>=3.8',
	install_requires=requirements
)
