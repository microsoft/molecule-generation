import os
import setuptools

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setuptools.setup(
    name="molecule_generation",
    use_scm_version=True,
    license="MIT",
    author="Krzysztof Maziarz",
    author_email="krzysztof.maziarz@microsoft.com",
    description="Implementations of deep generative models of molecules.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/microsoft/molecule-generation/",
    setup_requires=["setuptools_scm"],
    python_requires="==3.7.*",
    install_requires=[
        "dpu-utils>=0.2.13",
        "more-itertools",
        "numpy==1.19.2",  # Pinned due to incompatibility with `tensorflow`.
        "protobuf<4",  # Avoid the breaking 4.21.0 release.
        "scikit-learn>=0.24.1",
        "tensorflow==2.1.0",  # Pinned due to issues with `h5py`.
        "tf2_gnn>=2.13.0",
    ],
    packages=setuptools.find_packages(),
    package_data={"": ["test_datasets/*.pkl.gz", "test_datasets/*.smiles"]},
    include_package_data=True,
    entry_points={"console_scripts": ["molecule_generation = molecule_generation.cli.cli:main"]},
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
