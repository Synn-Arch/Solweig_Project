#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only
# Demo entrypoint (Dockerfile --target demo): run the API (:8000) and the
# studio frontend same-origin proxy (serve.py :8080) on one VM. The frontend
# process is the supervisor: if either process dies the container exits and
# fly restarts the machine (demo-grade supervision).
set -eu

mkdir -p /data/state /data/cache 2>/dev/null || true  # volume subdirs on first boot

cd /app/studio
python serve.py --host 0.0.0.0 --port 8080 --api http://127.0.0.1:8000 &
FRONTEND_PID=$!

cd /app
python -m solweig_gpu.server &
API_PID=$!

# Exit when EITHER process exits so fly's restart policy replaces the VM
# instead of serving a half-broken demo.
wait -n "$FRONTEND_PID" "$API_PID"
EXIT=$?
kill "$FRONTEND_PID" "$API_PID" 2>/dev/null || true
exit $EXIT
