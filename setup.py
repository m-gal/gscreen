"""
The final part is to create a setup.py file
for the custom Python package (called PROJECTNAME).

Because this is a package that is intended to stay local and not be uploaded to PyPI,
we only need to know its name and its version.
Everything else, including its description, long description,
author name, email address and more, are optional.
You can include it, but it isn't mandatory.

IMPORTANT:
FOR NOTEBOOK:
If you use a notebook you must add that cell at first:
    |   import sys
    |   sys.path.append('..')
FOR VS CODE:
If you use a VS Code Interactive Window you must add that cell at first:
    |   import os
    |   import sys
    |   sys.path.append('..')
    |
    |   try:
    |       import packagename
    |   except ModuleNotFoundError:
    |       os.chdir("../")

Alternatively, you need to run setup.py to install packagename
(every time you make a change to it).
Otherwise your notebooks won't see packagename (or its most recent version)


    Created on Dec 07 2020
    @author: mikhail.galkin
"""

#%% Creating setup.py
import setuptools


def readme():
    with open("README.md", "r") as f:
        return f.read()


setuptools.setup(
    name="gscreen",  # Required
    version="0.0.1",  # Required
    description="The ML home task from GreenScreen.ai",  # Optional
    long_description=readme(),  # Optional
    author="Mikhail Galkin",  # Optional
    author_email="mikhail_galkin@outlook.com",  # Optional
    packages=setuptools.find_packages(exclude=["tests"]),  # Required
    package_dir={"": "gscreen"},  # Optional
)
