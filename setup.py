import setuptools
import os

setuptools.setup(
    name="nionswift-structure-recognition",
    version="0.1",
    author="Jacob Madsen",
    author_email="jacob.madsen@univie.ac.at",
    description="Deep learning structure recognition of atomic resolution images",
    packages=["nionswift_plugin.nionswift_structure_recognition", "temnet", "fourier_scale_calibration"],
    package_data={"nionswift_plugin.nionswift_structure_recognition": ["models/*", "presets/*"]},
    include_package_data=True,
    python_requires='~=3.6',
)
