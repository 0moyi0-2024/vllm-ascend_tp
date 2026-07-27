#!/usr/bin/env python3
import json, urllib.request, concurrent.futures, sys, time
port=sys.argv[1]; par=int(sys.argv[2]); n=int(sys.argv[3])
url=f"http://127.0.0.1:{port}/v1/chat/completions"
prompts=["Count from 1 to 10.","What is the capital of France? Answer in one word.",
         "Say hello in Spanish.","Name a prime number.","What color is the sky?"]
def req(i):
    p={"model":"gemma4-eagle3","messages":[{"role":"user","content":prompts[i%len(prompts)]}],
       "max_tokens":48,"temperature":0}
    t0=time.time()
    try:
        r=urllib.request.urlopen(urllib.request.Request(url,data=json.dumps(p).encode(),
            headers={"Content-Type":"application/json"}),timeout=120)
        d=json.loads(r.read())
        c=d.get("choices",[{}])[0].get("message",{}).get("content","")
        return (time.time()-t0, c[:50], d.get("choices",[{}])[0].get("finish_reason"))
    except Exception as e:
        return (time.time()-t0, f"ERR {type(e).__name__}", None)
t0=time.time()
with concurrent.futures.ThreadPoolExecutor(par) as ex:
    res=list(ex.map(req, range(n)))
print(f"parallel={par} n={n} took={time.time()-t0:.1f}s")
for i,(dt,c,fr) in enumerate(res[:8]):
    print(f"  [{i}] {dt:.1f}s finish={fr} :: {c!r}")
