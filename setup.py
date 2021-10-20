import setuptools

from pathlib import Path

this_directory = Path(__file__).parent
with open(str(this_directory / "README.md"), encoding='utf-8') as f:
	long_description = f.read()

setuptools.setup(
	name='asrl-pyboreas',
	version='1.0.3',
	description='A toolkit for working with the Boreas dataset in Python',
	long_description=long_description,
	long_description_content_type='text/markdown',
	author='Keenan Burnett, Andrew Li, Shichen Lu, Jingxing Qian, Yuchen Wu, David Yoon',
	author_email='boreas@robotics.utias.utoronto.ca',
	url='https://github.com/utiasASRL/boreas-devkit',
	license='BSD',
	packages=setuptools.find_packages(),
	python_requires='>=3.8',
	install_requires=["numpy>=1.21.0",
					  "opencv-python>=4.5.3.56",
					  "matplotlib>=3.4.2",
					  "tqdm>=4.60.0",
					  "pyproj>=3.1.0",
					  "utm>=0.7.0",
					  "asrl-pysteam>=1.0.0"]
)
