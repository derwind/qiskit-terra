#!/bin/sh

python -B auto_typehints.py --qiskit-root qiskit --inplace quantum_info
#python -B auto_typehints.py --qiskit-root qiskit quantum_info --suffix .hinted --verbose --detect-missing-symbols
#python -B auto_typehints.py --qiskit-root qiskit quantum_info --suffix .hinted --only statevector.py one_qubit_decompose.py operator.py --verbose --detect-missing-symbols
