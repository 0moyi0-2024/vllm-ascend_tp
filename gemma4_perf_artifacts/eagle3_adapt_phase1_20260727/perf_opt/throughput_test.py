#!/usr/bin/env python3
import json, urllib.request, concurrent.futures, sys, time
port=sys.argv[1]; par=int(sys.argv[2]); n=int(sys.argv[3]); maxtok=int(sys.argv[4])
url=f"http://127.0.0.1:{port}/v1/completions"
# use /v1/completions with a fixed prompt, ignore_eos, to force maxtok decode
payload=lambda i:{"model":"gemma4-eagle3","prompt":f"The {i}-th story: Once upon a time",
   "max_tokens":maxtok,"temperature":0,"ignore_eos":True,"stream":False}
def req(i):
    t0=time.time()
    try:
        r=urllib.request.urlopen(urllib.request.Request(url,data=json.dumps(payload(i)).encode(),
            headers={"Content-Type":"application/json"}),timeout=600)
        d=json.loads(r.read())
        toks=d.get("usage",{}).get("completion_tokens",0)
        return (time.time()-t0, toks, d.get("choices",[{}])[0].get("finish_reason"))
    except Exception as e:
        return (time.time()-t0, 0, f"ERR {type(e).__name__}:{e}")
t0=time.time()
with concurrent.futures.ThreadPoolExecutor(par) as ex:
    res=list(ex.map(req, range(n)))
dt=time.time()-t0
tot=sum(t for _,t,_ in res)
print(f"parallel={par} n={n} maxtok={maxtok} wall={dt:.1f}s total_out_tokens={tot} agg_tok/s={tot/dt:.1f}")
ok=sum(1 for _,t,fr in res if t>0 and not str(fr).startswith("ERR"))
print(f"success={ok}/{n}; per-req avg tok={tot/max(ok,1):.0f}, avg wall/req={sum(t for t,_,_ in res)/max(ok,1):.1f}s")
