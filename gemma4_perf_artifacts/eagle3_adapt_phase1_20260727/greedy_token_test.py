#!/usr/bin/env python3
"""Greedy correctness test via /v1/chat/completions (gemma4 is a chat model).
Runs N rounds, checks determinism + sensible output. Captures text for spec-on/off diff."""
import json, sys, urllib.request, argparse

def chat(url, model, prompt, max_tokens):
    payload = {"model": model,
               "messages": [{"role": "user", "content": prompt}],
               "temperature": 0, "max_tokens": max_tokens, "seed": 0,
               "stream": False}
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        body = json.loads(r.read())
    msg = body["choices"][0].get("message", {})
    return msg.get("content",""), body["choices"][0].get("finish_reason"), body.get("usage")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="8000")
    ap.add_argument("--model", default="gemma4-target")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--out", default="/tmp/greedy_rounds.json")
    args = ap.parse_args()
    url = f"http://127.0.0.1:{args.port}/v1/chat/completions"
    prompts = [
        "What is the capital of France? Answer in one word.",
        "Count from 1 to 10.",
        "Explain what machine learning is in one sentence.",
        "Say hello in Spanish, French, and German.",
    ]
    rounds_out = []
    for rd in range(args.rounds):
        rd_out = []
        for p in prompts:
            text, fr, usage = chat(url, args.model, p, args.max_tokens)
            rd_out.append({"prompt": p, "text": text, "finish": fr, "usage": usage})
            print(f"[round {rd}] {p!r} -> {text[:90]!r} (finish={fr} tok={usage.get('completion_tokens') if usage else '-'})", flush=True)
        rounds_out.append(rd_out)
    stable = True
    for ri in range(1, len(rounds_out)):
        for pi, (a, b) in enumerate(zip(rounds_out[0], rounds_out[ri])):
            if a["text"] != b["text"]:
                stable = False
                print(f"MISMATCH prompt {pi} round0 vs round{ri}:\n  r0={a['text'][:80]!r}\n  r{ri}={b['text'][:80]!r}", flush=True)
    print(f"\nSTABILITY: {'PASS (all rounds identical)' if stable else 'FAIL'}", flush=True)
    json.dump(rounds_out, open(args.out,"w"), indent=2, ensure_ascii=False)
    print(f"saved {args.out}", flush=True)

if __name__ == "__main__":
    main()
