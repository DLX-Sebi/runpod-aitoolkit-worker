"""Ephemeral-pod trainer: boot → train → upload result JSON to S3 → self-terminate.

The site creates a pod whose start command fetches trainer_core.py + this file
and runs it. Job spec arrives base64-encoded in the JOB_JSON env var. The result
(same shape as the serverless handler's return + status) is written to
s3://<bucket>/<S3_PREFIX>/results/<job_id>.json — the site polls that object.

Self-termination is best-effort here AND unconditionally repeated by the start
command after python exits, so a crash can never leave the pod bleeding money.
"""

import base64
import json
import os
import sys
import time
import traceback

import trainer_core as core


def write_result(job_id, payload):
    c = core.s3()
    key = f"{core.S3_PREFIX}/results/{job_id}.json"
    c.put_object(Bucket=core.S3_BUCKET, Key=key,
                 Body=json.dumps(payload).encode(), ContentType="application/json")
    print(f"result written to {key}", flush=True)


def self_terminate():
    pod_id = os.environ.get("RUNPOD_POD_ID")
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not (pod_id and api_key):
        print("no RUNPOD_POD_ID/API_KEY — cannot self-terminate", flush=True)
        return
    import requests
    for _ in range(3):
        r = requests.delete(f"https://rest.runpod.io/v1/pods/{pod_id}",
                            headers={"Authorization": f"Bearer {api_key}"}, timeout=30)
        print(f"self-terminate HTTP {r.status_code}", flush=True)
        if r.ok:
            return
        time.sleep(5)


def main():
    raw = os.environ.get("JOB_JSON", "")
    if not raw:
        print("JOB_JSON missing", flush=True)
        return 1
    inp = json.loads(base64.b64decode(raw))
    job_id = inp.get("job_id") or "podjob"
    t0 = time.time()
    try:
        if inp.get("task") == "seed":
            out = core.seed_repos(inp.get("repos", []))
        else:
            out = core.run_train(inp, job_id, progress_cb=lambda m: print(m, flush=True))
        write_result(job_id, {"status": "COMPLETED", "output": out,
                              "gpu_seconds": round(time.time() - t0, 1)})
        return 0
    except Exception as e:
        traceback.print_exc()
        try:
            write_result(job_id, {"status": "FAILED", "error": str(e)[:4000],
                                  "gpu_seconds": round(time.time() - t0, 1)})
        except Exception:
            traceback.print_exc()
        return 1
    finally:
        self_terminate()


if __name__ == "__main__":
    sys.exit(main())
