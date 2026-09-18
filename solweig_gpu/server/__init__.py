# SPDX-License-Identifier: GPL-3.0-only
"""Optional HTTP API and persistence layer for the incremental design tool.

This subpackage is deliberately not imported by ``solweig_gpu`` itself: the
web framework stays optional for the core scientific package. Import
``solweig_gpu.server`` explicitly after installing the extra::

    pip install 'solweig-gpu[server]'
    from solweig_gpu.server import create_app

Importing this package without the ``server`` extra raises an ``ImportError``
with installation instructions. The persistence layer (``store``) and the
binary patch codec (``patch_codec``) have no FastAPI dependency.

Deployment note: the application ships without authentication or transport
security by design; it is meant to run behind an authenticating reverse proxy
on a trusted network (see the P9 deployment_operations plan).
"""

from solweig_gpu.server.store import (
    IdempotencyConflict,
    IdempotentReplay,
    JobRecord,
    ResultNotPublishable,
    ResultRecord,
    ScenarioQuotaExceeded,
    ScenarioRecord,
    SceneVersionConflict,
    StaleResultError,
    Store,
    StoreError,
    uv_to_world,
    world_to_uv,
)

SERVER_IMPORT_ERROR = (
    "the SOLWEIG design-tool API requires web dependencies that are not "
    "installed. Install them with: pip install 'solweig-gpu[server]' "
    "(fastapi, uvicorn, httpx, zstandard)"
)

try:  # server-extra modules, guarded so the core project never needs them.
    from solweig_gpu.server.patch_codec import (
        PATCH_MEDIA_TYPE,
        PATCH_SCHEMA_VERSION,
        PayloadChecksumError,
        PayloadCompressionError,
        PayloadDtypeError,
        PayloadLengthError,
        PayloadSchemaError,
        PayloadShapeError,
        PatchCodecError,
        build_manifest,
        decode_payload,
        encode_payload,
        payload_etag,
    )
    from solweig_gpu.server.app import RateLimiter, ScenarioQuotaLimiter, create_app
    from solweig_gpu.server.jobs import (
        JobRunner,
        RunnerContext,
        SiteConfig,
        SiteRegistry,
        SolveRequest,
        SolveResult,
        make_exact_worker_solver,
    )
    from solweig_gpu.server.models import (
        ApiError,
        CreateScenarioRequest,
        EditRequest,
        TreeObject,
    )
except ImportError as _error:  # pragma: no cover - extra not installed
    # ImportError covers both a missing distribution and blocked/stubbed
    # modules (e.g. ``sys.modules["fastapi"] = None``).
    raise ImportError(SERVER_IMPORT_ERROR) from _error

__all__ = [
    "ApiError",
    "CreateScenarioRequest",
    "EditRequest",
    "IdempotencyConflict",
    "IdempotentReplay",
    "JobRecord",
    "JobRunner",
    "PATCH_MEDIA_TYPE",
    "PATCH_SCHEMA_VERSION",
    "PayloadChecksumError",
    "PayloadCompressionError",
    "PayloadDtypeError",
    "PayloadLengthError",
    "PayloadSchemaError",
    "PayloadShapeError",
    "PatchCodecError",
    "RateLimiter",
    "ResultNotPublishable",
    "ResultRecord",
    "RunnerContext",
    "ScenarioQuotaLimiter",
    "ScenarioQuotaExceeded",
    "ScenarioRecord",
    "SceneVersionConflict",
    "SiteConfig",
    "SiteRegistry",
    "SolveRequest",
    "SolveResult",
    "StaleResultError",
    "Store",
    "StoreError",
    "TreeObject",
    "build_manifest",
    "create_app",
    "decode_payload",
    "encode_payload",
    "make_exact_worker_solver",
    "payload_etag",
    "uv_to_world",
    "world_to_uv",
]
