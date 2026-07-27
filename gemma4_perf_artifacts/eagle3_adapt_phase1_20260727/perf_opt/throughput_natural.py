#!/usr/bin/env python3
import json, urllib.request, concurrent.futures, sys, time
port=sys.argv[1]; par=int(sys.argv[2]); n=int(sys.argv[3]); model=sys.argv[4]
url=f"http://127.0.0.1:{port}/v1/chat/completions"
prompts=["Explain machine learning in detail.","Describe the history of computing in several sentences.",
         "Write a short paragraph about the ocean.","What are the benefits of exercise? Explain.",
         "Tell me about the solar system.","Explain how a computer works in detail.",
         "Describe the process of photosynthesis.","What is the importance of education? Explain."]
def req(i):
    p={"model":model,"messages":[{"role":"user","content":prompts[i%len(prompts)]}],
       "max_tokens":128,"temperature":0,"stream":False}
    t0=time.time()
    try:
        r=urllib.request.urlopen(urllib.request.Request(url,data=json.dumps(p).encode(),
            headers={"Content-Type":"application/json"}),timeout=300)
        d=json.loads(r.read())
        toks=d.get("usage",{}).get("completion_tokens",0)
        return (time.time()-t0, toks)
    except Exception as e:
        return (time.time()-t0, 0)
for PAR in [8,16]:
    t0=time.time()
    with concurrent.futures.ThreadPoolExecutor(PAR) as ex:
        res=list(ex.map(req, range(n)))
    dt=time.time()-t0; tot=sum(t for _,t in res); ok=sum(1 for _,t in res if t>0)
    print(f"spec-on model={model} parallel={PAR} n={n} wall={dt:.1f}s total_tok={tot} agg_tok/s={tot/dt:.1f} ok={ok}/{n}", flush=True)
