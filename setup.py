#!/usr/bin/env python

from setuptools import find_packages, setup

version = "0.1dev"

with open("README.md", encoding="UTF-8") as f:
    readme = f.read()

with open("requirements.txt", encoding="UTF-8") as f:
    required = f.read().splitlines()

setup(
    name="carmack",
    version=version,
    description="",
    long_description=readme,
    long_description_content_type="text/markdown",
    keywords=[],
    author="",
    author_email="",
    url="",
    license="",
    entry_points={"console_scripts": ["carmack=carmack.__main__:run_carmack"]},
    python_requires=">=3.10",
    install_requires=required,
    packages=find_packages(exclude="docs"),
    zip_safe=False,
    include_package_data=True,
    package_data={
        'carmack': ['data/*']
    }
)
