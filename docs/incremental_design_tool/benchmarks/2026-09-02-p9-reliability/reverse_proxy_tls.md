# Reverse-proxy TLS termination (REL-003)

The API process itself never terminates TLS (deployment_operations.md: "HTTPS
in public deployment" via "reverse proxy (optional)"). Terminate TLS at a
reverse proxy in front of uvicorn and let the app see the real client through
the standard forwarded headers. This requires running uvicorn with
`--proxy-headers` (the smoke harness `tests/test_deployment_smoke.py` boots
exactly this configuration and proves the per-IP rate limiter keys on
`X-Forwarded-For`).

Application server (one process, forwarded headers ON). The app is built
with `create_app(...)` (see `solweig_gpu/server/app.py`); a minimal launcher
module makes it addressable to uvicorn:

```python
# serve.py
from pathlib import Path

from solweig_gpu.server.app import create_app

app = create_app(
    state_root=Path("/opt/solweig-app/state"),
    sites={"campus-1km-v1": {"cache_dir": Path("/opt/solweig-app/cache/campus-1km-v1")}},
)
```

```bash
uvicorn serve:app \
    --host 127.0.0.1 --port 8000 \
    --proxy-headers --forwarded-allow-ips 127.0.0.1
```

## Caddy

Caddy terminates TLS with automatic certificates. SSE needs no extra
directives (Caddy disables proxy response buffering by default and does not
time out long-lived streams within the app's own `timeout_seconds`):

```text
design.example.edu {
    reverse_proxy 127.0.0.1:8000
}
```

## nginx

nginx buffers proxied responses by default, which would hold SSE events until
the proxy buffer fills — the stream would look dead to the classroom
frontend. Disable buffering on the events route (the app also sends
`X-Accel-Buffering: no`, which nginx honors per-response):

```nginx
server {
    listen 443 ssl;
    server_name design.example.edu;

    ssl_certificate     /etc/letsencrypt/live/design.example.edu/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/design.example.edu/privkey.pem;

    client_max_body_size 2m;   # > app-level 1 MiB cap so the app's 413 wins

    # Large/payload downloads: sendfile-friendly, no buffering concerns.
    location /api/v1/scenarios/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # SSE: buffering off, generous read timeout (app closes the stream itself
    # at timeout_seconds, default 300s).
    location ~ ^/api/v1/scenarios/[^/]+/events$ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 3600s;
        chunked_transfer_encoding on;
    }

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## Notes

- `X-Forwarded-For` must reach uvicorn, and `--forwarded-allow-ips` must be
  restricted to the proxy address; otherwise clients could spoof their rate
  budget by sending their own `X-Forwarded-For`.
- The app-level controls (1 MiB body cap, per-IP 30 req/min, per-scenario
  20 edits/min) are in-process and per-API-process. Multi-process deployments
  (one uvicorn per core behind the proxy) need proxy-level rate limiting
  because the in-process buckets do not aggregate.
- Health endpoints `/health/live`, `/health/ready`, `/health/worker` and
  `/metrics` are exempt from the app-level rate limits, so uptime probes can
  always reach the process.
