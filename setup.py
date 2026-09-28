#!/bin/env python
# -*- coding: utf-8 -*-
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
# Copyright (C) {2025}  {Tobias Sebastian Finn}
#
# Monthly adaptation: Andrew Atkinson, 2026 (frazil, based on GenSIM)

from setuptools import find_packages, setup

setup(
    # Distribution name differs from GenSIM's (and from the unrelated NLP
    # package "gensim" on PyPI); the import package stays `gensim`.
    name='frazil',
    packages=find_packages(
        include=["gensim"]
    ),
    version='0.1.0',
    description='Unofficial monthly adaptation of the GenSIM sea-ice model (Finn et al., 2025)',
    author='Andrew Atkinson (adaptation); Tobias Finn (original GenSIM)',
    url='https://github.com/andrew-atkinson/frazil',
    license='MIT',
)
