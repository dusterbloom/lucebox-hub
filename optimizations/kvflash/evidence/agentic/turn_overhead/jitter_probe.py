# host-stall detector: sleeps 10 ms in a loop, reports any gap > 150 ms (monotonic and wall clocks). No server, no GPU.
import time, sys
dur=float(sys.argv[1]) if len(sys.argv)>1 else 100
t0=time.monotonic(); last=t0; lastw=time.time()
while time.monotonic()-t0<dur:
    time.sleep(0.01)
    n=time.monotonic(); nw=time.time()
    if n-last>0.15: print('t=%6.1fs stall %.2fs (wall-clock delta %.2fs)'%(n-t0,n-last,nw-lastw),flush=True)
    last=n; lastw=nw
print('done')
