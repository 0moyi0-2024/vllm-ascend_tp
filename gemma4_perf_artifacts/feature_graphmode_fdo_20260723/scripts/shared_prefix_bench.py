#!/usr/bin/env python3
"""Shared-prefix workload: long identical system prompt + varied short question.
Demonstrates automatic prefix caching (APC) benefit via reduced TTFT.
Runs two arms: (A) shared prefix, (B) no shared prefix (random), reports TTFT."""
import argparse, json, time, random, urllib.request, statistics, concurrent.futures

def post(url, payload, timeout=120):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = json.loads(r.read())
        ttft = time.time() - t0
        ok = "choices" in body
        return ttft, ok
    except Exception as e:
        return time.time() - t0, False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prefix-len", type=int, default=4096)
    ap.add_argument("--num-prompts", type=int, default=80)
    ap.add_argument("--request-rate", type=float, default=8)
    args = ap.parse_args()
    url = f"http://127.0.0.1:{args.port}/v1/chat/completions"
    # Build a long shared prefix (repeated filler text ~ prefix-len tokens)
    filler = ("The quick brown fox jumps over the lazy dog. " * 200)
    shared_prefix = filler[:args.prefix_len * 4]  # ~4 chars/token
    questions = ["What is 2+2?", "Summarize the above.", "Name a prime number.",
                 "Translate hello to French.", "What color is the sky?"]

    def run_arm(shared: bool):
        results = []
        interval = 1.0 / args.request_rate
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as ex:
            futs = []
            for i in range(args.num_prompts):
                if shared:
                    content = shared_prefix + "\n\nQuestion: " + random.choice(questions)
                else:
                    content = filler[:random.randint(200, args.prefix_len*4)] + "\n\nQuestion: " + random.choice(questions)
                payload = {"model": args.model, "messages":[{"role":"user","content":content}],
                           "max_tokens": 32, "temperature": 0}
                futs.append(ex.submit(post, url, payload))
                time.sleep(interval)
            for f in futs:
                results.append(f.result())
        ttfts = [t for t, ok in results if ok]
        return ttfts, sum(1 for _,ok in results if ok)

    print(f"=== Arm A: SHARED prefix (len~{args.prefix_len}) ===", flush=True)
    a_ttfts, a_ok = run_arm(True)
    print(f"ok={a_ok}/{args.num_prompts} ttft_mean={statistics.mean(a_ttfts):.4f} "
          f"p50={statistics.median(a_ttfts):.4f} p95={sorted(a_ttfts)[int(len(a_ttfts)*0.95)]:.4f}", flush=True)
    print(f"=== Arm B: NO shared prefix (random) ===", flush=True)
    b_ttfts, b_ok = run_arm(False)
    print(f"ok={b_ok}/{args.num_prompts} ttft_mean={statistics.mean(b_ttfts):.4f} "
          f"p50={statistics.median(b_ttfts):.4f} p95={sorted(b_ttfts)[int(len(b_ttfts)*0.95)]:.4f}", flush=True)
    if a_ttfts and b_ttfts:
        print(f"=== TTFT reduction (shared vs random): "
              f"{(1-statistics.mean(a_ttfts)/statistics.mean(b_ttfts))*100:.1f}% ===", flush=True)

if __name__ == "__main__":
    main()
