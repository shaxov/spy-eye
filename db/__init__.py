from .memory import MemoryDB


def initialize(kind, params=None):
    if kind == 'memory':
        return MemoryDB()
    else:
        raise ValueError(f"Database of kind '{kind}' is not found.")
