#!/bin/sh

python -B auto_typehints.py --qiskit-root qiskit --inplace quantum_info
#python -B auto_typehints.py --qiskit-root qiskit quantum_info --suffix .hinted
