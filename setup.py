"""Code developement for reservoir models."""
import re
from setuptools import setup, find_packages

with open('deepfield/__init__.py', 'r') as f:
    VERSION = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
    if not VERSION:
        raise RuntimeError("Unable to find version string.")
    VERSION = VERSION.group(1)

with open('docs/index.rst', 'r') as f:
    LONG_DESCRIPTION = f.read()

setup(
    name='GeoCode',
    packages=find_packages(exclude=['docs']),
    version=VERSION,
    url='https://github.com/geo-kit/GeoCode',
    license='Apache License 2.0',
    author='geo-kit',
    author_email='',
    description='Code developement for reservoir models',
    long_description=LONG_DESCRIPTION,
    zip_safe=False,
    platforms='any',
    install_requires=[
        'setuptools',
		'numpy',
		'chardet',
		'pandas >= 2.1.0',
		'scikit-learn',
		'scikit-image',
		'anytree',
		'ipywidgets',
		'pyvista',
		'tqdm',
		'numba',
		'tables',
		'pytest',
		'psutil',
		'dask',
		'georead @ git+https://github.com/geo-kit/georead'
    ],
    extras_require={
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: Apache License 2.0',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.13',
        'Topic :: Scientific/Engineering'
    ],
)
