# Fly.io demo deployment (cheapest shape)

Pre-deploy of the deployable baseline while the realtime mission runs. Single
`shared-cpu-1x` (1 shared vCPU / 2 GB) machine, scale-to-zero: idle cost
≈ disk only (~$1-2/mo), pay-per-demo-minute when used — inside a $14/mo
ceiling. (Trial orgs cannot launch `performance-1x` machines; upgrade later if
edit latency matters for a demo.) Replaced at mission close (R8) by the
commissioning deploy.

## What's deployed

- **API** `python -m solweig_gpu.server` on 127.0.0.1:8000 (env-configured
  boot, `SOLWEIG_*` per deployment_operations.md)
- **Studio frontend** `studio/serve.py` on :8080,
  same-origin `/api` reverse proxy (fly terminates HTTPS at :443 → :8080)
- **State + site cache** share ONE named volume `solweig_disk` (3 GB) at
  `/data` — fly machines mount at most one volume per machine, so state lives
  at `/data/state` and the site cache at `/data/cache/site_500` (uploaded
  out-of-band; NEVER baked into the image — data policy)

Demo target = `Dockerfile --target demo` (derives from the same `runtime`
stage; base targets and CMD unchanged for the compose shape).

## Honest demo limits

- **Cold start**: machine suspended at zero → first request pays ~10 s boot +
  ~30-60 s numba JIT + cache mmap. Subsequent requests warm.
- **Exact edit latency**: measured 38-40 s on the M1 dev host; expect roughly
  50-70 s on performance-1x amd64 (single dedicated vCPU). Demo-grade.
- **Realtime collaboration**: R1 operation surface (`POST /workspaces/*/operations`)
  is present; SSE events/epochs land with the r1-epochs merge — until then
  the studio demo is the exact-edit experience.

## One-time provisioning

```sh
flyctl auth login
flyctl apps create solweig-rt-demo        # globally unique; mirror in fly.demo.toml
flyctl volumes create solweig_disk --size 3 -r nrt   # ONE volume max per machine
flyctl deploy -c fly.demo.toml --build-target demo
./deploy/fly-demo/upload_site_cache.sh solweig-rt-demo
flyctl machine restart -a solweig-rt-demo
```

## Smoke check

The fly edge routes to the studio proxy (:8080), not the API, so `/health/*`
404s — the proxied capability document is the probe (it also serves as the
readiness signal once the site cache is uploaded and the JIT warm-up is done):

```sh
curl -fsS https://solweig-rt-demo.fly.dev/api/v1/capabilities | head
```

Then open `https://solweig-rt-demo.fly.dev/` in a browser (first load is the
cold start; scenario creation materializes from baseline_results in ~0.5 s).

## Teardown / redeploy

```sh
flyctl deploy -c fly.demo.toml --build-target demo   # new image, volumes persist
flyctl apps destroy solweig-rt-demo              # full teardown
```
