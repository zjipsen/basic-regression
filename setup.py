# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='basic-regression',
    version='1.0.0',
    description='Use basic regression methods to analyze data',
    long_description=readme,
    author='Zana Jipsen',
    author_email='zjipsen@gmail.com',
    url='https://github.com/zjipsen/basic-regression',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)
