import os
from setuptools import setup, find_packages

path = os.path.abspath(os.path.dirname(__file__))

try:
    with open(os.path.join(path, 'README.md'), "r", encoding='utf-8') as f:
        long_description = f.read()
except Exception as e:
    long_description = "scBiG: a novel scRNA-seq representation learning method based on graph node embedding"

setup(
    name="scbig",
    version="0.1.0",
    keywords=["single-cell RNA-sequencing", "Graph node embedding", "Dimensionality reduction"],
    description="scBiG: a novel scRNA-seq representation learning method based on graph node embedding",
    long_description=long_description,
    long_description_content_type='text/markdown',
    license="MIT Licence",

    url="https://github.com/sldyns/scBiG",
    author="Kun Qian, Ting Li",
    author_email="kun_qian@foxmail.com",
    maintainer="Kun Qian",
    maintainer_email="kun_qian@foxmail.com",

    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy",
        "scanpy",
        "h5py",
        "torch",
        "dgl",
        "pandas",
        "scipy",
        "sklearn",
        "louvain"
        ],
    platforms='any'
)