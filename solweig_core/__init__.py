"""solweig_core: numerical-profile identity and runtime certification.

T01 scope: this package pins *which* numerical semantics produced a result
(profile ids) and certifies that the solweig_gpu origin actually imported (or
claimed by an artifact) still hashes to the recorded sources. It contains no
scientific kernels; importing it must not import torch.
"""
from .profile import (
    CANONICAL_CPU_V1,
    LEGACY_CUDA_V1,
    PROFILE_REGISTRY,
    KEY_SOURCES,
    ProfileManifest,
    capture_imported_origin,
    snapshot_tree,
    certify,
)

__all__ = [
    "CANONICAL_CPU_V1",
    "LEGACY_CUDA_V1",
    "PROFILE_REGISTRY",
    "KEY_SOURCES",
    "ProfileManifest",
    "capture_imported_origin",
    "snapshot_tree",
    "certify",
]
