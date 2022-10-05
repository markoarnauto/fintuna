from setuptools import setup, find_packages

setup(
    name="fintuna",
    packages=find_packages(),
    version="0.1.0",
    author="Cortecs GmbH",
    author_email="markus.tretzmueller@cortecs.ai",
    description="Parameter optimization for finance",
    url="https://github.com/markoarnauto/fintuna",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "setuptools>=61.0",
        "pandas~=1.2.3",
        "numpy~=1.20.0",
        "scikit-learn==0.24.1",
        "APScheduler==3.7.0",
        "lightgbm==3.2.1",
        "optuna~=3.0.0",
        "tables~=3.7.0",
        "myst-nb~=0.17.1"
    ],
)