"""
Attack components for the LIBRA design space.

    framing_registry   the 43 framing candidates and the dedupe pass
    libra_framings     the 12 UTOS templates and the composition operators
    obfuscation_v2     struct_* framings
    (plinius_* framings now live in third_party/l1b3rt4s/, AGPL-3.0)

Submodules are imported directly rather than re-exported here: the registry
resolves framings by name at call time, and a package-level import of one
family would make the others load unnecessarily.
"""
