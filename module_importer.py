import os
import sys
import argparse
import glob
import importlib
from typing import List


def import_module(file_path):
    file_path, _ = os.path.splitext(file_path)  # e.g. path/to/qiskit/quantum_info/states/statevector
    path_elems = file_path.split(os.sep)  # e.g. ['path', 'to', 'qiskit', 'quantum_info', 'states', 'statevector']
    loc = path_elems.index('qiskit')
    module_name = '.'.join(path_elems[loc:])

    return importlib.import_module(module_name)


def import_modules(module_name: str, qiskit_root: str, only_filename: List[str] | None = None, verbose: bool = False):
    module_root = None
    for file_path in glob.glob(os.path.join(qiskit_root, '**/'), recursive=True):
        if os.path.isdir(file_path):
            if file_path[-1] == os.sep:
                file_path = file_path[:-1]
            if file_path.split(os.sep)[-1] == module_name:
                module_root = file_path
                break

    if module_root is None:
        raise ModuleNotFoundError(f"'{module_name}' is not found")

    sys.path.append(qiskit_root)
    sys.path.append(module_root)

    failed_files = []
    for file_path in glob.glob(os.path.join(module_root, '**/*.py'), recursive=True):
        if only_filename and os.path.basename(file_path) not in only_filename:
            continue

        try:
            mod = import_module(file_path)
            if verbose:
                print(mod)
        except Exception as e:
            print(e)
            failed_files.append(file_path)

    if failed_files:
        print('-' * 50)
        print('[Summary for failed files]')
        for file_path in failed_files:
            print(file_path)
    else:
        if verbose:
            print('-' * 50)
            print('[Summary]')
            print('No error!')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--qiskit-root', dest='qiskit_root', type=str, default='qiskit', help='/path/to/qiskit_root')
    parser.add_argument('--only', dest='only_filename', nargs='*', type=str, default=[], help='file name')
    parser.add_argument('--verbose', action='store_true', help='output logs?')
    parser.add_argument('module_name', type=str, help="module_name such as 'quantum_info'")

    args = parser.parse_args()

    return args


def main():
    args = parse_opt()
    import_modules(args.module_name, args.qiskit_root, args.only_filename, args.verbose)


if __name__ == '__main__':
    main()
