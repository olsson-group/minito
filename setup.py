from setuptools import setup, find_packages

setup(
    name="minito",
    version="0.0.1",
    description="minimal ito library for rapid prototyping",
    author="Simon Olsson",
    author_email="simonols@chalmers.se",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)