# attacks/__init__.py
"""
Attack implementations
"""

from .obfuscation import LexicalObfuscator, StructuralObfuscator, create_obfuscation_suite

__all__ = ['LexicalObfuscator', 'StructuralObfuscator', 'create_obfuscation_suite']