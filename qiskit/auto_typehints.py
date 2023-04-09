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
from collections import OrderedDict
from typing import Dict, Tuple


if sys.version_info.major < 3 or sys.version_info.minor < 10:
    info = sys.version_info
    raise NotImplementedError(f'Python {info.major}.{info.minor}.{info.micro} is not supported')


class SignatureImprover:
    def __init__(self, file_path: str, module_name: str = 'quantum_info', verbose: bool = False):
        file_path, _ = os.path.splitext(file_path)
        path_elems = file_path.split(os.sep)
        loc = path_elems.index(module_name)
        self.module_name = '.'.join(path_elems[loc + 1 :])
        self.mod = importlib.import_module(self.module_name)
        self.verbose = verbose
        self._methods2signatures = None

    @property
    def methods2signatures(self):
        return self._methods2signatures

    def run(self) -> Dict[str, str]:
        if self._methods2signatures is not None:
            return self._methods2signatures

        methods2signatures = {}
        for x in inspect.getmembers(self.mod):
            if not isinstance(x, tuple):
                continue
            obj_name, obj_type = x
            if inspect.isclass(obj_type):
                m2th = self._class_proc(self.module_name, obj_name, obj_type)
                methods2signatures.update(m2th)

        self._methods2signatures = methods2signatures
        return self._methods2signatures

    def _class_proc(self, module_name: str, short_class_name: str, class_type) -> Dict[str, str]:
        methods2signatures = {}
        full_class_name = self.extract_class_name(str(class_type))
        if self.isclass_in_file(full_class_name, module_name):
            # sig = inspect.signature(class_type)
            # print(class_type, '->', sig)
            for y in inspect.getmembers(class_type):
                if not isinstance(y, tuple):
                    continue
                member_name, member_type = y
                if inspect.isfunction(member_type):
                    method_name, signature = self._method_proc(short_class_name, member_name, member_type)
                    if method_name is not None:
                        methods2signatures[method_name] = signature

        return methods2signatures

    def _method_proc(self, class_name: str, short_method_name: str, method_type) -> Tuple[str, str] | Tuple[None, None]:
        full_method_name = str(method_type).split(' ')[1]
        if not self.directly_belongs_to(full_method_name, class_name):
            return None, None

        signature = inspect.signature(method_type)
        if self.verbose:
            print('[Type Hint]', full_method_name, '=>', signature.parameters, '->', signature.return_annotation)

        # print('[DOCSTRING]', inspect.getdoc(method_type))
        docstring = docstring_parser.google.parse(inspect.getdoc(method_type))
        arg_types = OrderedDict({arg.arg_name: arg.type_name for arg in docstring.params})
        returns_types = docstring.returns
        arg_types = self.modernize_type_infos(arg_types)
        if self.verbose:
            print('[DOCSTRING]', arg_types, '->', returns_types.args[-1] if returns_types else '')

        return short_method_name, self._supplement_signature(signature, arg_types, returns_types)

    def _supplement_signature(
        self, signature: inspect.Signature, doc_arg_types: OrderedDict[str, str], doc_returns_types: str | inspect._empty
    ) -> str:
        name2hint = OrderedDict()  # key: qualified_name
        name2default = OrderedDict()  # key: qualified_name

        for name, detail in signature.parameters.items():
            # e.g., '**kwargs'
            qualified_name = re.split(r'\s*:\s*', re.split(r'\s*=\s*', str(detail))[0])[0]

            # no type hint
            if detail.annotation == inspect.Parameter.empty:
                hint = None
                if name in doc_arg_types:
                    hint = doc_arg_types[name]
                name2hint[qualified_name] = hint
            else:
                name2hint[qualified_name] = str(detail.annotation)
            # no default value
            if detail.default == inspect.Parameter.empty:
                name2default[qualified_name] = inspect.Parameter.empty
            else:
                # type hint doesn't suggest 'nullable' but default value does
                if detail.default is None and name2hint[qualified_name] is not None:
                    if 'Optional' not in name2hint[qualified_name] and 'None' not in name2hint[qualified_name]:
                        name2hint[qualified_name] += ' | None'
                name2default[qualified_name] = detail.default

        new_signature_elems = []
        for name, hint in name2hint.items():
            if hint is not None:
                name_with_info = f'{name}: {hint}'
            else:
                name_with_info = name
            if name2default[name] != inspect.Parameter.empty:
                name_with_info += f' = {name2default[name]}'
            new_signature_elems.append(name_with_info)
        new_signature = '(' + ', '.join(new_signature_elems) + ')'

        if signature.return_annotation != inspect.Parameter.empty:  # from type hint
            new_signature += f' -> {self.extract_class_name(str(signature.return_annotation))}'
        else:
            if doc_returns_types:  # from docstring
                if isinstance(doc_returns_types, docstring_parser.common.DocstringReturns):
                    new_signature += f' -> {doc_returns_types.type_name}'
                elif inspect.isclass(doc_returns_types):
                    full_class_name = self.extract_class_name(str(doc_returns_types))
                    new_signature += f' -> {full_class_name}'
                else:
                    new_signature += f' -> {str(doc_returns_types)}'

        if self.verbose:
            print('[Signature]', new_signature)
            print()

        return new_signature

    @staticmethod
    def extract_class_name(class_xxx: str):
        if m := re.search(r"'(\S+)'", class_xxx):
            return m.group(1)
        else:
            return class_xxx

    @staticmethod
    def isclass_in_file(class_name: str, module_name: str) -> bool:
        return '.'.join(class_name.split('.')[:-1]) == module_name

    @staticmethod
    def directly_belongs_to(method_name: str, class_name: str) -> bool:
        return method_name.split('.')[0] == class_name

    @staticmethod
    def modernize_type_infos(arg_types: Dict[str, str]) -> Dict[str, str]:
        def improve_hints(types: str | None):
            if types is None:
                return None
            return re.sub(r'\s+or\s+', ' | ', types)

        return {k: improve_hints(v) for k, v in arg_types.items()}


class SignatureReplacer:
    def __init__(self, file_path: str, signature_improver: SignatureImprover, out_file_path: str | None = None):
        self.file_path = file_path
        self.signature_improver = signature_improver
        self.out_file_path = out_file_path

    def run(self):
        fout = sys.stdout
        if self.out_file_path is not None:
            fout = open(self.out_file_path, 'w')

        with open(self.file_path) as fin:
            class_name = None
            method_name = None
            for line in fin.readlines():
                line = line.rstrip()

                # end of class definition
                if class_name is not None and re.search(r'^\S', line):
                    class_name = None

                # end of method signature
                if method_name is not None and ':' in line:
                    method_name = None

                if m := re.search(r'^class\s+(\S+)\s*\(', line):
                    class_name = m.group(1)
                    print(line, file=fout)
                    continue

                if class_name is not None:
                    if m := re.search(r'^\s+def\s+(\S+)\s*\(', line):
                        method_name = m.group(1)
                        if method_name in self.signature_improver.methods2signatures:
                            signature = self.signature_improver.methods2signatures[method_name]
                            loc = line.index('def')
                            print(f'{" " * loc}def {method_name}{signature}:', file=fout)
                        else:
                            # not improvements, just output
                            print(line, file=fout)

                        if ':' in line:
                            method_name = None  # forget immediately
                    else:
                        if method_name is None:
                            print(line, file=fout)
                else:
                    print(line, file=fout)

        if fout != sys.stdout:
            fout.close()


def autohints(target_dir: str, out_file: str | None = None, verbose: bool = False):
    for file_path in glob.glob(os.path.join(target_dir, '**/*.py'), recursive=True):
        if os.path.basename(file_path) == 'statevector.py':
            signature_improver = SignatureImprover(file_path, verbose=verbose)
            signature_improver.run()

            if verbose:
                for method_name, signature in signature_improver.methods2signatures.items():
                    print(f'{method_name}{signature}')

            signature_replacer = SignatureReplacer(file_path, signature_improver, out_file_path=out_file)
            signature_replacer.run()

            break


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', dest='out_file', type=str, default=None, help='/path/to/file')
    parser.add_argument('--verbose', action='store_true', help='output logs?')
    parser.add_argument('dir', type=str, help='/path/to/directory')

    args = parser.parse_args()

    return args


def main():
    args = parse_opt()
    autohints(args.dir, args.out_file, args.verbose)


if __name__ == '__main__':
    main()
