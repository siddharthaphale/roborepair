"Canonical program representations for structural comparisons and row metadata."

from __future__ import annotations

import ast
import hashlib

# Preserve robot API names when normalizing identifiers and extracting call sequences.
API_NAMES = (
    "get_object_pose",
    "get_gripper_pose",
    "goto_pose",
    "open_gripper",
    "close_gripper",
    "check_grasped",
    "execute_waypoints",
    "lift",
    "place_on",
    "pull",
)


class _Canonicalizer(ast.NodeTransformer):
    """Rename user identifiers to canonical v0, v1, ... (API names untouched)."""

    def __init__(self, api_names: frozenset[str]):
        self.api = api_names
        self.map: dict[str, str] = {}

    def _canon(self, name: str) -> str:
        if name in self.api:
            return name
        if name not in self.map:
            self.map[name] = f"v{len(self.map)}"
        return self.map[name]

    def visit_Name(self, node: ast.Name):
        return ast.copy_location(ast.Name(id=self._canon(node.id), ctx=node.ctx), node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        node.name = self._canon(node.name)
        self.generic_visit(node)
        return node

    def visit_arg(self, node: ast.arg):
        node.arg = self._canon(node.arg)
        return node

    def visit_Constant(self, node: ast.Constant):
        if isinstance(node.value, float):
            return ast.copy_location(ast.Constant(value=round(node.value, 3)), node)
        return node


def _strip_docstrings(tree: ast.AST) -> ast.AST:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module, ast.ClassDef)):
            body = getattr(node, "body", [])
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                node.body = body[1:] or [ast.Pass()]
    return tree


def normalized_ast_dump(code: str, api_names=API_NAMES) -> str | None:
    """Canonical AST dump of ``code`` or None if it does not parse."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return None
    tree = _strip_docstrings(tree)
    tree = _Canonicalizer(frozenset(api_names)).visit(tree)
    return ast.dump(tree, annotate_fields=False, include_attributes=False)


def normalized_ast_hash(code: str, api_names=API_NAMES) -> str | None:
    dump = normalized_ast_dump(code, api_names)
    if dump is None:
        return None
    return hashlib.sha256(dump.encode()).hexdigest()[:16]
