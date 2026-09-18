# solweig-core-cpu — the torch-free runtime

The runtime replay distribution of the solweig ultrafast bitwise plan
(T14, DESIGN 5.4): numba CPU kernels + frozen-site-capture loaders +
fallback router + the general step-table producer, packaged WITHOUT the
oracle stack (torch), the preprocess stack (GDAL/rasterio), plotting,
notebook IO, or any C++ AOT toolchain.

The runtime executes exactly one backend, `numba-full-solve`, over
cached captures. Oracle-only operations refuse with typed errors
(`OracleBackendRefusal`, `CaptureRegenerationRefusal`,
`UnseenSceneRefusal`, `profile.OracleEnvironmentError`) — there is never
a silent torch import.

Oracle work (capture generation, regeneration after geometry edits,
unseen-scene amplitude policies, the legacy torch backend) belongs to the
repo-root dev install and its environment, never to this one.
