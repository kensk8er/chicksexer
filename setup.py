# -*- coding: UTF-8 -*-
import pip
from setuptools import setup
from codecs import open
from os import path
import re

__author__ = 'kensk8er1017@gmail.com'

PACKAGE_NAME = 'chicksexer'


def get_version():
    """get version number"""
    version_file = path.join(PACKAGE_NAME, '_version.py')
    version_string = open(version_file, "rt").read()
    match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_string, re.M)

    if match:
        version = match.group(1)
    else:
        raise RuntimeError("Unable to find version string in %s." % (version_file,))

    return version


current_path = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(current_path, PACKAGE_NAME, 'README.md'), encoding='utf-8') as file_:
    long_description = file_.read()

# Add dependencies into install_requires if they are on PyPI
requirements = open('requirements.txt', 'r').read().splitlines()
install_requires = list()
for requirement in requirements:
    if requirement.startswith('git+'):
        # install_requires doesn't work for dependency starting from git+
        pip.main(['install', requirement])
    else:
        install_requires.append(requirement)

setup(
    name=PACKAGE_NAME,
    version=get_version(),
    description='Python package for gender classification.',
    long_description=long_description,
    url='https://github.com/kensk8er/chicksexer',
    author='Kensuke Muraki',
    author_email='kensk8er1017@gmail.com',
    maintainer='Kensuke Muraki',
    maintainer_email='kensk8er1017@gmail.com',
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',
        "Programming Language :: Python",
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Text Processing :: Linguistic',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    packages=[PACKAGE_NAME],
    install_requires=install_requires,
    zip_safe=False,
    keywords='natural-language-processing machine-learning tensorflow deep-learning recurrent-neural-networks lstm nlp python neural-network character-embeddings data-science gender-classification',
    include_package_data=True,
)
