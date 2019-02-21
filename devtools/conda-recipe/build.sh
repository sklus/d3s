#!/usr/bin/env bash

export BOOST=${PREFIX}
python setup.py install --single-version-externally-managed --record record.txt
