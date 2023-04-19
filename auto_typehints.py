import os
import sys
import re
import argparse
import glob
import importlib
import inspect
import ast
from dataclasses import dataclass
import docstring_parser
from collections import OrderedDict
from typing import Dict, Tuple


if sys.version_info.major < 3 or sys.version_info.minor < 10:
    info = sys.version_info
    raise NotImplementedError(f'Python {info.major}.{info.minor}.{info.micro} is not supported')


@dataclass(eq=False)
class AstClassInfo:
    module: str
    alias: str | None


class ImportVisitor(ast.NodeVisitor):
    def __init__(self):
        self._modules: Dict[str, str] = {}  # import numpy as np : numpy -> np
        self._name2info: Dict[str, AstClassInfo] = {}

    @property
    def modules(self) -> Dict[str, str]:
        return self._modules

    @property
    def name2info(self) -> Dict[str, AstClassInfo]:
        return self._name2info

    def visit_Import(self, node):
        for alias in node.names:
            self._modules[alias.name] = alias.asname

    def visit_ImportFrom(self, node):
        module = node.module or ''
        for alias in node.names:
            self._name2info[alias.name] = AstClassInfo(module, alias.name)


class ClassInfo:
    def __init__(self, module_name: str, short_class_name: str, class_type, visitor: ImportVisitor, verbose: bool = False) -> Dict[str, str]:
        self.visitor = visitor
        self.verbose = verbose
        self._methods2signatures = {}
        full_class_name = self.extract_class_name(str(class_type))
        if self.isclass_in_file(full_class_name, module_name):
            # sig = inspect.signature(class_type)
            # print(class_type, '->', sig)
            for y in inspect.getmembers(class_type):
                if not isinstance(y, tuple):
                    continue
                member_name, insp_member = y
                # normal method or bound method
                if inspect.isfunction(insp_member) or inspect.ismethod(insp_member):
                    is_classmethod = False
                    if member_name in class_type.__dict__:
                        # https://stackoverflow.com/questions/19947151/how-to-distinguish-an-instance-method-a-class-method-a-static-method-or-a-func
                        is_classmethod = isinstance(class_type.__dict__[member_name], classmethod)
                    method_name, signature = self._method_proc(short_class_name, member_name, insp_member, is_classmethod)
                    if method_name is not None:
                        self._methods2signatures[method_name] = signature

    @property
    def methods2signatures(self):
        return self._methods2signatures

    def _method_proc(self, short_class_name: str, short_method_name: str, insp_method, is_classmethod: bool) -> Tuple[str, str] | Tuple[None, None]:
        if inspect.isfunction(insp_method):
            full_method_name = str(insp_method).split(' ')[1]
        elif inspect.ismethod(insp_method):
            full_method_name = str(insp_method).split(' ')[2]
        else:
            raise NotImplementedError(f'Not supported: {insp_method}')

        if not self.directly_belongs_to(full_method_name, short_class_name):
            return None, None

        signature = inspect.signature(insp_method)
        if self.verbose:
            print('[Type Hint]', full_method_name, '=>', signature.parameters, '->', signature.return_annotation)

        # print('[DOCSTRING]', inspect.getdoc(method_type))
        try:
            # may contain invalid docstring
            docstring = docstring_parser.google.parse(inspect.getdoc(insp_method))
        except:
            return None, None
        arg_types = OrderedDict({arg.arg_name: arg.type_name for arg in docstring.params})
        returns_types = docstring.returns
        arg_types = self.modernize_type_infos(arg_types)
        if self.verbose:
            ok = returns_types and returns_types.args[-1] not in self.ignore_cases
            print('[DOCSTRING]', arg_types, '->', returns_types.args[-1] if ok else '')

        new_signature = self._supplement_signature(signature, arg_types, returns_types, short_class_name, is_classmethod)
        if self.verbose:
            print(f'[Signature] {short_method_name}{new_signature}')
            print()

        return short_method_name, new_signature

    def _supplement_signature(
        self,
        signature: inspect.Signature,
        doc_arg_types: OrderedDict[str, str],
        doc_returns_types: str | inspect._empty,
        short_class_name: str,
        is_classmethod: bool,
    ) -> str:
        def fix_hint(hint, class_name: str = short_class_name, visitor: ImportVisitor = self.visitor):
            if hint is None:
                return None

            hint_parts = []
            for h in re.split(r'\s*\|\s*', hint):
                if h == class_name:
                    hint_parts.append(f"'{h}'")
                elif h == 'string':
                    hint_parts.append('str')
                else:
                    parts = h.split('.')
                    if len(parts) > 1:
                        module, name = '.'.join(parts[:-1]), parts[-1]
                        # omit needless decorations
                        module = module.replace('~', '')
                        # override h if needed
                        if module in visitor.modules:
                            module = visitor.modules[module]
                            h = '.'.join([module, name])
                        elif module.find('qiskit.') < 0 and 'qiskit.' + module in visitor.modules:
                            module = visitor.modules['qiskit.' + module]
                            h = '.'.join([module, name])
                        elif name in visitor.name2info:
                            info = visitor.name2info[name]
                            mod = importlib.import_module(info.module)
                            mod2 = importlib.import_module(module)
                            try:
                                mod3 = importlib.import_module('qiskit.' + module)
                            except:
                                mod3 = None
                            if mod3 is not None and getattr(mod, name) == getattr(mod3, name):
                                if info.alias is None:
                                    h = name
                                else:
                                    h = info.alias
                            elif getattr(mod, name) == getattr(mod2, name):
                                if info.alias is None:
                                    h = name
                                else:
                                    h = info.alias
                    hint_parts.append(h)
            return ' | '.join(hint_parts)

        name2hint = OrderedDict()  # key: qualified_name
        name2default = OrderedDict()  # key: qualified_name

        if is_classmethod:
            # For some reason, in the class method case, 'cls' is omitted in the 'signature', so I add it.
            name2hint['cls'] = None
            name2default['cls'] = inspect.Parameter.empty

        for name, detail in signature.parameters.items():
            # e.g., '**kwargs'
            qualified_name = re.split(r'\s*:\s*', re.split(r'\s*=\s*', str(detail))[0])[0]

            # no type hint
            if detail.annotation == inspect.Parameter.empty:
                hint = None
                if name in doc_arg_types:
                    hint = doc_arg_types[name]
                name2hint[qualified_name] = fix_hint(hint)
            else:
                name2hint[qualified_name] = fix_hint(str(detail.annotation))
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
        # arguments of methods
        for name, hint in name2hint.items():
            # assign type hint to given variable
            if hint is not None:
                name_with_info = f'{name}: {hint}'
            else:
                name_with_info = name
            # assign default value to given variable
            if name2default[name] != inspect.Parameter.empty:
                name_with_info += f' = {name2default[name]}'
            new_signature_elems.append(name_with_info)
        new_signature = '(' + ', '.join(new_signature_elems) + ')'

        # returns of methods
        if signature.return_annotation != inspect.Parameter.empty:  # from type hint
            return_type = fix_hint(self.extract_class_name(str(signature.return_annotation)))
            new_signature += f' -> {return_type}'
        else:
            if doc_returns_types:  # from docstring
                if isinstance(doc_returns_types, docstring_parser.common.DocstringReturns):
                    if doc_returns_types.type_name not in self.ignore_cases:
                        new_signature += f' -> {fix_hint(doc_returns_types.type_name)}'
                elif inspect.isclass(doc_returns_types):
                    full_class_name = self.extract_class_name(str(doc_returns_types))
                    new_signature += f' -> {fix_hint(full_class_name)}'
                else:
                    return_type = fix_hint(str(doc_returns_types))
                    new_signature += f' -> {return_type}'

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

    @classmethod
    def modernize_type_infos(cls, arg_types: Dict[str, str]) -> Dict[str, str]:
        def improve_hints(types: str | None):
            if types is None or types in cls.ignore_cases:
                return None
            return re.sub(r'\s+or\s+', ' | ', types)

        return {k: improve_hints(v) for k, v in arg_types.items()}

    ignore_cases = frozenset({'CLASS'})


class SignatureImprover:
    def __init__(self, file_path: str, visitor: ImportVisitor, verbose: bool = False):
        file_path, _ = os.path.splitext(file_path)  # e.g. path/to/qiskit/quantum_info/states/statevector
        path_elems = file_path.split(os.sep)  # e.g. ['path', 'to', 'qiskit', 'quantum_info', 'states', 'statevector']
        loc = path_elems.index('qiskit')
        self.module_name = '.'.join(path_elems[loc:])  # e.g. 'qiskit.quantum_info.states.statevector'
        self.mod = importlib.import_module(self.module_name)
        self.visitor = visitor
        self.verbose = verbose
        self._classname2info: Dict[str, ClassInfo] = {}

    @property
    def class_names(self):
        return self._classname2info.keys()

    def methods2signatures(self, class_name: str):
        if class_name in self._classname2info:
            return self._classname2info[class_name].methods2signatures
        else:
            return None

    def run(self) -> None:
        """Construct a new signature for the method after constructing the type hint from docstring and return it as a dictionary from the method name."""
        for x in inspect.getmembers(self.mod):
            if not isinstance(x, tuple):
                continue
            obj_name, obj_type = x
            if inspect.isclass(obj_type):
                info = ClassInfo(self.module_name, obj_name, obj_type, self.visitor, self.verbose)
                self._classname2info[obj_name] = info


class SignatureReplacer:
    """Update signatures of the methods in the given file and output the updated file.

    Args:
        file_path (str): /path/to/file
        signature_improver (SignatureImprover): SignatureImprover instance which has already run
        suffix (str | None): suffix of file or stdout
    """

    def __init__(self, file_path: str, signature_improver: SignatureImprover, suffix: str | None = None, inplace: bool = False):
        self.file_path = file_path
        self.signature_improver = signature_improver
        self.suffix = suffix
        self.inplace = inplace

    def run(self):
        fout = sys.stdout
        if self.inplace:
            import tempfile

            _, out_file_path = tempfile.mkstemp()
            fout = open(out_file_path, 'w')
        elif self.suffix is not None:
            dirname = os.path.dirname(self.file_path)
            basename, ext = os.path.splitext(os.path.basename(self.file_path))
            out_file_path = os.path.join(dirname, f'{basename}{self.suffix}{ext}')
            fout = open(out_file_path, 'w')

        with open(self.file_path) as fin:
            class_name = None
            method_name = None
            new_method_decl = None
            for line in fin.readlines():
                line = line.rstrip()

                # end of class definition
                if class_name is not None and re.search(r'^\S', line):
                    class_name = None

                # end of method signature
                if method_name is not None and ':' in line:
                    if new_method_decl is not None:
                        print(new_method_decl, file=fout)
                        new_method_decl = None
                    method_name = None
                    continue

                if m := re.search(r'^class\s+(\S+)\s*\(', line):
                    class_name = m.group(1)
                    print(line, file=fout)
                    continue

                if class_name is not None:
                    if m := re.search(r'^\s+def\s+(\S+)\s*\(', line):
                        method_name = m.group(1)
                        if method_name in self.signature_improver.methods2signatures(class_name):
                            signature = self.signature_improver.methods2signatures(class_name)[method_name]
                            loc = line.index('def')
                            new_method_decl = f'{" " * loc}def {method_name}{signature}:'
                        else:
                            # not improvements, just output
                            print(line, file=fout)

                        if ':' in line:
                            if new_method_decl is not None:
                                print(new_method_decl, file=fout)
                                new_method_decl = None
                            method_name = None  # forget immediately
                    else:
                        if method_name is None:
                            print(line, file=fout)
                else:
                    print(line, file=fout)

        if fout != sys.stdout:
            fout.close()

        if self.inplace:
            import shutil

            shutil.move(out_file_path, self.file_path)


def autohints(
    module_name: str, qiskit_root: str, suffix: str | None = None, inplace: bool = False, only_filename: str | None = None, verbose: bool = False
):
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

    for file_path in glob.glob(os.path.join(module_root, '**/*.py'), recursive=True):
        if suffix is not None and file_path.endswith(f'{suffix}.py'):
            continue
        if only_filename is not None and os.path.basename(file_path) != only_filename:
            continue

        try:
            with open(file_path) as fin:
                module = ast.parse(fin.read())

            visitor = ImportVisitor()
            visitor.visit(module)

            if verbose:
                print(visitor.modules)
                print(visitor.name2info)

            signature_improver = SignatureImprover(file_path, visitor, verbose=verbose)
            signature_improver.run()

            if verbose:
                print('Signatures per class')
                for class_name in signature_improver.class_names:
                    print(f'[{class_name}]')
                    for method_name, signature in signature_improver.methods2signatures(class_name).items():
                        print(f'  {method_name}{signature}')

            signature_replacer = SignatureReplacer(file_path, signature_improver, suffix=suffix, inplace=inplace)
            signature_replacer.run()
        except Exception as e:
            print('[[Exception]]', file_path, e, file=sys.stderr)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--qiskit-root', dest='qiskit_root', type=str, default='qiskit', help='/path/to/qiskit_root')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--suffix', dest='suffix', type=str, default=None, help='suffix of file')
    group.add_argument('--inplace', action='store_true', help='in-place replacement?')
    parser.add_argument('--only', dest='only_filename', type=str, default=None, help='file name')
    parser.add_argument('--verbose', action='store_true', help='output logs?')
    parser.add_argument('module_name', type=str, help="module_name such as 'quantum_info'")

    args = parser.parse_args()

    return args


def main():
    args = parse_opt()
    autohints(args.module_name, args.qiskit_root, args.suffix, args.inplace, args.only_filename, args.verbose)


if __name__ == '__main__':
    main()
