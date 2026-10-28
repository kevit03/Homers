# HOMERs runbook

Live: https://homers-nu.vercel.app (Vercel project `homers`, Hobby plan, deployed from the laptop by `./deploy.sh`).

## Deploy
- `./deploy.sh --preview` → preview URL; check it, then `./deploy.sh` → production.
- `./deploy.sh --rebuild` re-exports dashboard.html from the newest model first.
- Always deploy through `deploy.sh` (it runs `export/build_site.py`). A bare `vercel deploy --cwd site` ships whatever
  stale `site/` is on disk. That is how /api/record went missing in the Oct 26, 2026 deploy.
- Other Claude sessions may be editing dashboard_template.html; make sure dashboard.html is a finished build before deploying.

## After a deploy, check
```
curl -s -o /dev/null -w "%{http_code}\n" https://homers-nu.vercel.app/            # 200
curl -s -o /dev/null -w "%{http_code}\n" https://homers-nu.vercel.app/api/record  # 200 (502 = ESPN or Tempo failed)
curl -sI https://homers-nu.vercel.app/ | grep -i x-content-type-options           # nosniff
```

## Roll back
- Fastest: `npx vercel rollback --cwd site` (back to the previous production deployment), or in the dashboard:
  vercel.com → homers → Deployments → pick the last good one → ⋯ → Promote to Production / Instant Rollback.
- Rolling back does not touch the repo; fix forward with `./deploy.sh` afterwards.

## Where things are
- Function logs (/api/record): vercel.com → homers → Logs. Hobby keeps 1 hour, so check soon after a problem.
- Build/deploy history: vercel.com → homers → Deployments.
- Usage (function invocations, CPU, bandwidth): vercel.com → Usage. Hobby doesn't bill for overage; it pauses the
  feature until the 30-day window resets.
- CI: GitHub → kevit03/Homers → Actions (secret scan + syntax checks on push).

## Alerts
- Who gets them: the repo owner (GitHub emails on failed workflow runs).
- Uptime: none yet (see .ship/status.json next_steps).
