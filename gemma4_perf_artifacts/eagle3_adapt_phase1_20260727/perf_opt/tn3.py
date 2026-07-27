#!/usr/bin/env python3
import json, urllib.request, concurrent.futures, sys, time
port=sys.argv[1]; model=sys.argv[2]; n=int(sys.argv[3])
url=f"http://127.0.0.1:{port}/v1/chat/completions"
prompts=["Explain machine learning in detail.","Describe the history of computing.","Write a paragraph about the ocean.","Benefits of exercise?","Tell me about the solar system.","How a computer works.","Describe photosynthesis.","Importance of education?"]
def req(i):
    p={"model":model,"messages":[{"role":"user","content":prompts[i%len(prompts)]}],"max_tokens":128,"temperature":0,"stream":False}
    t0=time.time()
    try:
        r=urllib.request.urlopen(urllib.request.Request(url,data=json.dumps(p).encode(),headers={"Content-Type":"application/json"}),timeout=300)
        d=json.loads(r.read()); return (time.time()-t0, d.get("usage",{}).get("completion_tokens",0))
    except Exception: return (time.time()-t0, 0)
for PAR in [1,2,4]:
    t0=time.time()
    with concurrent.futures.ThreadPoolExecutor(PAR) as ex: res=list(ex.map(req, range(n)))
    dt=time.time()-t0; tot=sum(t for _,t in res)
    print(f"model={model} parallel={PAR} n={n} wall={dt:.1f}s agg_tok/s={tot/dt:.1f}", flush=True)
