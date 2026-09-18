# SOLWEIG-GPU incremental design tool — single-node CPU API image.
#
# Research-level deployment shape (see
# docs/incremental_design_tool/deployment_operations.md "Container deployment"):
#   * API + in-process worker thread, no GPU, no distributed queue;
#   * state directory mounted read-write at /state;
#   * site caches bind/volume mounted READ-ONLY at /site-cache
#     (SOLWEIG_CACHE_ROOT/site_id must exist there — never baked in);
#   * frontend served separately (studio/serve.py
#     or any static server + same-origin proxy) — see docker-compose.yml.

# --- Builder: compile the osgeo GDAL bindings (no Linux wheels exist). ----
FROM python:3.14-slim AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

# osgeo bindings must match the Debian libgdal version; torch comes from the
# CPU-only wheel index so the CUDA libraries never touch this deployment.
RUN pip wheel --no-cache-dir --wheel-dir /wheels \
        "gdal==$(gdal-config --version).*" && \
    pip wheel --no-cache-dir --wheel-dir /wheels torch \
        --index-url https://download.pytorch.org/whl/cpu

# --- Runtime ---------------------------------------------------------------
FROM python:3.14-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Runtime shared libraries: rasterio's bundled GDAL dlopens libexpat, and the
# compiled osgeo bindings link against Debian's libgdal runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libexpat1 libgdal36 \
    && rm -rf /var/lib/apt/lists/*

# Install the prebuilt wheels via a bind mount so they never enter a layer.
# torch is satisfied here, so `pip install .[server]` below cannot pull the
# multi-GB CUDA-enabled default wheel.
RUN --mount=type=bind,from=builder,source=/wheels,target=/wheels \
    pip install --no-cache-dir /wheels/*

COPY . .
RUN pip install --no-cache-dir .[server] \
    && rm -rf /app/build /app/solweig_gpu.egg-info

# Non-root runtime. /state is pre-created with the runtime user's ownership
# so a named volume mounted there inherits it on first use.
RUN useradd --create-home --uid 10001 solweig \
    && mkdir -p /state \
    && chown -R solweig:solweig /state
USER solweig

EXPOSE 8000

CMD ["python", "-m", "solweig_gpu.server"]

# --- Demo target (single-VM shape: API + same-origin frontend proxy). ------
# Base targets keep the compose/classroom shape (API only, CMD above). The
# demo target derives from the SAME runtime stage — no layer duplication —
# and only swaps the entrypoint for fly.io scale-to-zero demos
# (deploy/fly-demo/). Build with: --target demo
FROM runtime AS demo
COPY --chown=solweig:solweig deploy/fly-demo/entrypoint-demo.sh /entrypoint-demo.sh
RUN chmod +x /entrypoint-demo.sh
EXPOSE 8080
CMD ["/entrypoint-demo.sh"]
