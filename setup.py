import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

dependencies = ['rpy2', 'statsmodels', 'pandas', 'numpy', 'matplotlib',
		'sklearn', 'tqdm', 'pylab', 'pickle', 'db_queries', 'db_tools', 'scipy']

setuptools.setup(
    name="ensemble",
    version="0.0.1",
    author="Marlena Bannick",
    author_email="mnorwood@uw.edu",
    description="Simple Ensemble Models for Global Health Epidemiology",
    long_description=long_description,
    long_description_content_type="text/markdown",
    include_package_data=True,
    install_requirements=dependencies,
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 2",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    )
)


