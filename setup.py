#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages
from os import path


def read(rel_path):
    here = path.abspath(path.dirname(__file__))
    with open(path.join(here, rel_path), 'r') as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['Click>=7.0', 'click-log',
                'pyDOE2']

test_requirements = ['pytest>=3', ]

setup(
    author="Simen Eldevik",
    author_email='simen.eldevik@dnv.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Python package (Reciprocal Data and Physics models - RaPiD-models) to support more specific, accurate and timely decision support in operation of safety-critical systems, by combining physics-based modelling with data-driven machine learning and probabilistic uncertainty assessment.",
    entry_points={
        'console_scripts': [
            'rapid_models=rapid_models.cli:main',
        ],
    },
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='rapid_models',
    name='rapid_models',
    packages=find_packages(where='src',
                           exclude=['contrib', 'docs', 'tests'],
                           include=['rapid_models',
                                    'rapid_models.*']),
    package_dir={'': 'src'},
    test_suite='tests',
    tests_require=test_requirements,
    url='https://dnvgl-one.visualstudio.com/GRD%20Maritime/_git/rapid_models',
    version=get_version("src/rapid_models/__init__.py"),
    zip_safe=False,
)
