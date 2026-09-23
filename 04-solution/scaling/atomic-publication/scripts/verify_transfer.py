"""Assert that the numerical batch AST and all protected inputs are unchanged."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--repo', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
repo = args.repo.resolve()


def git(*arguments):
    return subprocess.check_output(['git', '-C', str(repo), *arguments], text=True)


class RemovePublication(ast.NodeTransformer):
    def visit_ImportFrom(self, node):
        return None if node.module == 'batch_safety' else node

    def visit_With(self, node):
        context = node.items[0].context_expr
        if isinstance(context, ast.Call) and isinstance(context.func, ast.Name) and context.func.id == 'atomic_output':
            return [self.visit(statement) for statement in node.body]
        return self.generic_visit(node)

    def visit_Name(self, node):
        if node.id == 'out_dir':
            return ast.Attribute(value=ast.Name(id='args', ctx=ast.Load()), attr='out_dir', ctx=node.ctx)
        return node

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == 'args.out_dir.mkdir':
            return None
        return self.generic_visit(node)


batch_path = '04-solution/service/app/batch.py'
before = ast.parse(git('show', f'7800f22:{batch_path}'))
after = ast.parse((repo / batch_path).read_text())
assert ast.dump(RemovePublication().visit(before)) == ast.dump(RemovePublication().visit(after))
safety_path = '04-solution/service/app/batch_safety.py'
source = (repo / safety_path).read_text()
original = git('show', f'guard/memory:{safety_path}')
assert source[source.index('def _check_destination'):] == original[original.index('def _check_destination'):]
module = ast.parse(source)
functions = [n.name for n in module.body if isinstance(n, ast.FunctionDef)]
assert functions == ['_check_destination', '_publish', 'atomic_output']
assert not any(x in source for x in ['estimate_memory', 'available_memory', 'check_memory', '/proc', 'cgroup'])
protected = ['04-solution/service/app/core', '04-solution/service/model',
             '04-solution/split', '04-solution/eval']
assert not git('diff', '7800f22', '--', *protected).strip()
assert git('rev-parse', 'guard/memory').strip() == 'de1c68cf97b23ad9732f6da06dcf693495618676'
result = {'baseline': git('rev-parse', '7800f22').strip(),
          'guard': git('rev-parse', 'guard/memory').strip(),
          'numerical_batch_ast_unchanged': True, 'publication_bodies_identical_to_guard': True,
          'no_memory_preflight': True, 'protected_paths_unchanged': protected,
          'functions': functions,
          'files': {p: hashlib.sha256((repo / p).read_bytes()).hexdigest() for p in [batch_path, safety_path]}}
args.output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
