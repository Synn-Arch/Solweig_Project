#!/bin/sh
# SPDX-License-Identifier: GPL-3.0-only
# One-time out-of-band data upload to the fly demo machine (data policy: site
# data never enters an image; it lives only on the volume). Two payloads:
#
#   1. site-cache/site_500 (~504 MB)  -> /data/cache/site_500
#      The incremental solve cache (mmap'd arrays, baseline results).
#   2. Input_subset/processed_inputs raster set (~34 MB, the tile's source
#      rasters EXCLUDING the redundant incremental_cache_0_0 scratch) ->
#      /data/site_dirs/site_500
#      The exact lane needs real rasters no cache carries: the oracle-location
#      derivation opens Building_DSM/Building_DSM_0_0.tif from the site dir
#      (solver._oracle_location), and forcing reads metfiles/metfile_0_0.txt.
#      Without them every exact job refuses ("site building raster missing").
#
# Usage: ./deploy/fly-demo/upload_site_cache.sh [app-name]
set -eu

APP="${1:-solweig-rt-demo}"
SITE_ID="site_500"
CACHE_ROOT="/data/cache"
SITE_DIR_ROOT="/data/site_dirs"
SOURCE_ROOT="Input_subset/processed_inputs"
# Raster families the runtime reads from a site dir (manifest-relative
# layout: <kind>/<kind>_<tile>.tif + metfiles/). incremental_cache_0_0 is
# deliberately absent — /data/cache already carries the real cache.
RASTER_KINDS="Building_DSM DEM Trees walls aspect metfiles"

if [ ! -f "site-cache/${SITE_ID}/manifest.json" ]; then
  echo "!! run from the repo root (site-cache/${SITE_ID} not found)" >&2
  exit 1
fi
if [ ! -d "${SOURCE_ROOT}/Building_DSM" ]; then
  echo "!! ${SOURCE_ROOT}/Building_DSM not found (run from repo root)" >&2
  exit 1
fi

echo "==> packing ${SITE_ID} cache"
CACHE_TARBALL="/tmp/solweig-${SITE_ID}-cache.tar.gz"
tar czf "$CACHE_TARBALL" -C site-cache "$SITE_ID"

echo "==> packing ${SITE_ID} source rasters (${RASTER_KINDS})"
RASTER_TARBALL="/tmp/solweig-${SITE_ID}-rasters.tar.gz"
# shellcheck disable=SC2086  # RASTER_KINDS is a word list by construction
tar czf "$RASTER_TARBALL" -C "$SOURCE_ROOT" $RASTER_KINDS

echo "==> uploading to ${APP} ($(du -h "$CACHE_TARBALL" | cut -f1) cache + $(du -h "$RASTER_TARBALL" | cut -f1) rasters)"
flyctl ssh sftp shell -a "$APP" <<EOF
  put ${CACHE_TARBALL} /tmp/cache.tar.gz
  put ${RASTER_TARBALL} /tmp/rasters.tar.gz
EOF

echo "==> extracting onto the volume"
# NB: `flyctl ssh console -C` execs argv directly (no shell) — one command
# per invocation, no shell builtins or && chains.
flyctl ssh console -a "$APP" -C "mkdir -p ${CACHE_ROOT} ${SITE_DIR_ROOT}" || exit 1
flyctl ssh console -a "$APP" -C "tar xzf /tmp/cache.tar.gz -C ${CACHE_ROOT}" || exit 1
flyctl ssh console -a "$APP" -C "mkdir -p ${SITE_DIR_ROOT}/${SITE_ID}" || exit 1
flyctl ssh console -a "$APP" -C "tar xzf /tmp/rasters.tar.gz -C ${SITE_DIR_ROOT}/${SITE_ID}" || exit 1
flyctl ssh console -a "$APP" -C "rm /tmp/cache.tar.gz /tmp/rasters.tar.gz" || exit 1
flyctl ssh console -a "$APP" -C "chown -R solweig:solweig ${CACHE_ROOT} ${SITE_DIR_ROOT}" || true
flyctl ssh console -a "$APP" -C "ls -la ${CACHE_ROOT}/${SITE_ID}"
flyctl ssh console -a "$APP" -C "ls ${SITE_DIR_ROOT}/${SITE_ID}/Building_DSM"

echo "==> done. Point SOLWEIG_SITE_DIR_${SITE_ID} at ${SITE_DIR_ROOT}/${SITE_ID}"
echo "    and SOLWEIG_SELECTED_DATE_STR_${SITE_ID} at the cache's build date,"
echo "    then restart the machine: flyctl machine restart -a ${APP}"
