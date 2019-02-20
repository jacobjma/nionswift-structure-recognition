import setuptools
import os

setuptools.setup(
    name="nionswift-structure-recognition",
    version="0.1",
    author="Jacob Madsen",
    author_email="jacob.madsen@univie.ac.at",
    description="Deep learning structure recognition of atomic resolution images",
    #url="",
    packages=["nionswift_plugin.nionswift_structure_recognition"],
    #package_data={"nion.eels_analysis": ["resources/*"]},
    #install_requires=["nionswift>=0.14.0"],
    #classifiers=[
    #    "Development Status :: 2 - Pre-Alpha",
    #    "Programming Language :: Python :: 3.6",
    #],
    include_package_data=True,
    python_requires='~=3.6',
)
