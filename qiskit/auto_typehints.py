import os
import sys
import re
sys.path.append('..')
sys.path.append('quantum_info')
import argparse
import glob
import importlib
import inspect
#from typing import get_type_hints
from docstring_parser import parse


if sys.version_info.major < 3 or sys.version_info.minor < 10:
    info = sys.version_info
    raise NotImplementedError(f'Python {info.major}.{info.minor}.{info.micro} is not supported')


def isclass_in_file(class_name: str, module_name: str) -> bool:
    return '.'.join(class_name.split('.')[:-1]) == module_name


def check(file_path):
    file_path, _ = os.path.splitext(file_path)
    path_elems = file_path.split(os.sep)
    loc = path_elems.index('quantum_info')
    module_name = '.'.join(path_elems[loc+1:])
    mod = importlib.import_module(module_name)

    for x in inspect.getmembers(mod):
        if not isinstance(x, tuple):
            continue
        obj_name, obj_type = x
        if inspect.isclass(obj_type):
            class_name = None
            if m := re.search(r"'(\S+)'", str(obj_type)):
                class_name = m.group(1)
            if isclass_in_file(class_name, module_name):
                print(obj_name)
                sig = inspect.signature(obj_type)
                print(sig)
                for y in inspect.getmembers(obj_type):
                    if not isinstance(y, tuple):
                        continue
                    member_name, member_type = y
                    if inspect.isfunction(member_type):
                        print(y)


def autohints(target_dir):
    for file_path in glob.glob(os.path.join(target_dir, '**/*.py'), recursive=True):
        if os.path.basename(file_path) == 'statevector.py':
            check(file_path)
            break


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=str, help='/path/to/directory')

    args = parser.parse_args()

    return args


def main():
    args = parse_opt()
    autohints(args.dir)


if __name__ == '__main__':
    main()
