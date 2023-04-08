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
from typing import Dict


if sys.version_info.major < 3 or sys.version_info.minor < 10:
    info = sys.version_info
    raise NotImplementedError(f'Python {info.major}.{info.minor}.{info.micro} is not supported')


class Checker:
    def __init__(self, file_path: str, module_name: str = 'quantum_info'):
        file_path, _ = os.path.splitext(file_path)
        path_elems = file_path.split(os.sep)
        loc = path_elems.index(module_name)
        self.module_name = '.'.join(path_elems[loc + 1 :])
        self.mod = importlib.import_module(self.module_name)

    def run(self):
        for x in inspect.getmembers(self.mod):
            if not isinstance(x, tuple):
                continue
            obj_name, obj_type = x
            if inspect.isclass(obj_type):
                self._class_proc(self.module_name, obj_name, obj_type)

    def _class_proc(self, module_name: str, short_class_name: str, class_type):
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
                    self._method_proc(short_class_name, member_name, member_type)

    def _method_proc(self, class_name: str, short_method_name: str, method_type):
        full_method_name = str(method_type).split(' ')[1]
        if self.directly_belongs_to(full_method_name, class_name):
            sig = inspect.signature(method_type)
            print(full_method_name, '=>', sig)

            # print('[DOCSTRING]', inspect.getdoc(method_type))
            docstring = docstring_parser.google.parse(inspect.getdoc(method_type))
            arg_types = {arg.arg_name: arg.type_name for arg in docstring.params}
            arg_types = self.improve_type_infos(arg_types)
            print('[DOCSTRING]', arg_types)
            print()

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
            ...
        return {k: improve_hints(v) for k, v in arg_types.items()}


def autohints(target_dir):
    for file_path in glob.glob(os.path.join(target_dir, '**/*.py'), recursive=True):
        if os.path.basename(file_path) == 'statevector.py':
            Checker(file_path).run()
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
