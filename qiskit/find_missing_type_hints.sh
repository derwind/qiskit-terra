#!/bin/sh

mypy --strict quantum_info/ | grep "Function is missing a type annotation"
