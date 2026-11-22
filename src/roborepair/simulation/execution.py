"""Syntax checks shared by subprocess-based environment executors."""


def compile_program(code: str) -> tuple[bool, str | None]:
    """Report compilation success and a readable syntax error without executing code."""
    try:
        compile(code, "<program>", "exec")
        return True, None
    except SyntaxError as exc:
        return False, f"{type(exc).__name__}: {exc.msg} (line {exc.lineno})"
