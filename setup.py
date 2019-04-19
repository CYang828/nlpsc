# coding=utf-8


import sys
from setuptools import setup, find_packages


kwargs = {}
install_requires = []
version = '1.0.0'

if sys.version_info < (3, 0):
    with open('README.md') as f:
        kwargs['long_description'] = f.read()

    with open('requirements.txt') as f:
        for require in f:
            install_requires.append(require[:-1])
elif sys.version_info > (3, 0):
    with open('README.md', encoding='utf-8') as f:
        kwargs['long_description'] = f.read()

    with open('requirements.txt', encoding='utf-8') as f:
        for require in f:
            install_requires.append(require[:-1])

kwargs['install_requires'] = install_requires

setup(
    name='nlpsc',
    version=version,
    include_package_data=True,
    packages=find_packages(),
    package_data={'nlpsc': ['default/stopwords.txt']},
    entry_points={},
    author='bslience',
    description="nlpsc is a util for nlp",
    **kwargs
)
