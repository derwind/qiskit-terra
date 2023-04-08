import os
import sys
import re

sys.path.append('..')
sys.path.append('quantum_info')
import argparse
import glob
import importlib
import inspect
import docstring_parser
from typing import Dict, Tuple


if sys.version_info.major < 3 or sys.version_info.minor < 10:
    info = sys.version_info
    raise NotImplementedError(f'Python {info.major}.{info.minor}.{info.micro} is not supported')


class Checker:
    def __init__(self, file_path: str, module_name: str = 'quantum_info', verbose: bool = False):
        file_path, _ = os.path.splitext(file_path)
        path_elems = file_path.split(os.sep)
        loc = path_elems.index(module_name)
        self.module_name = '.'.join(path_elems[loc + 1 :])
        self.mod = importlib.import_module(self.module_name)
        self.verbose = verbose

    def run(self) -> Dict[str, str]:
        methods2typehints = {}
        for x in inspect.getmembers(self.mod):
            if not isinstance(x, tuple):
                continue
            obj_name, obj_type = x
            if inspect.isclass(obj_type):
                m2th = self._class_proc(self.module_name, obj_name, obj_type)
                methods2typehints.update(m2th)
        return methods2typehints

    def _class_proc(self, module_name: str, short_class_name: str, class_type) -> Dict[str, str]:
        methods2typehints = {}
        full_class_name = None
        if m := re.search(r"'(\S+)'", str(class_type)):
            full_class_name = m.group(1)
        if self.isclass_in_file(full_class_name, module_name):
            # sig = inspect.signature(class_type)
            # print(class_type, '->', sig)
            for y in inspect.getmembers(class_type):
                if not isinstance(y, tuple):
                    continue
                member_name, member_type = y
                if inspect.isfunction(member_type):
                    method_name, typehints = self._method_proc(short_class_name, member_name, member_type)
                    if method_name is None:
                        methods2typehints[method_name] = typehints
        return methods2typehints

    def _method_proc(self, class_name: str, short_method_name: str, method_type) -> Tuple[str | None, str | None]:
        full_method_name = str(method_type).split(' ')[1]
        if not self.directly_belongs_to(full_method_name, class_name):
            return None, None

        signature = inspect.signature(method_type)
        if self.verbose:
            print(full_method_name, '=>', signature)

        # print('[DOCSTRING]', inspect.getdoc(method_type))
        docstring = docstring_parser.google.parse(inspect.getdoc(method_type))
        arg_types = {arg.arg_name: arg.type_name for arg in docstring.params}
        returns_types = docstring.returns
        arg_types = self.improve_type_infos(arg_types)
        if self.verbose:
            print('[DOCSTRING]', arg_types, '->', returns_types.args[-1] if returns_types else 'None')
            print()

        return full_method_name, arg_types

    @staticmethod
    def isclass_in_file(class_name: str, module_name: str) -> bool:
        return '.'.join(class_name.split('.')[:-1]) == module_name

    @staticmethod
    def directly_belongs_to(method_name: str, class_name: str) -> bool:
        return method_name.split('.')[0] == class_name

    @staticmethod
    def improve_type_infos(arg_types: Dict[str, str]) -> Dict[str, str]:
        def improve_hints(types: str | None):
            if types is None:
                return None
            return re.sub(r'\s+or\s+', ' | ', types)

        return {k: improve_hints(v) for k, v in arg_types.items()}


def autohints(target_dir: str, verbose: bool = False):
    for file_path in glob.glob(os.path.join(target_dir, '**/*.py'), recursive=True):
        if os.path.basename(file_path) == 'statevector.py':
            Checker(file_path, verbose=verbose).run()
            break


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', action='store_true', help='output logs?')
    parser.add_argument('dir', type=str, help='/path/to/directory')

    args = parser.parse_args()

    return args


def main():
    args = parse_opt()
    autohints(args.dir, args.verbose)


if __name__ == '__main__':
    main()
