import re, statistics, sys

log = open(sys.argv[1]).read().splitlines()
tot = [float(m.group(1)) for l in log for m in [re.search(r"gpu scorer: .* total ([\d.]+) ms", l)] if m]
unav = [l for l in log if "gpu scorer unavailable" in l]
resel = [l for l in log if "[kvflash] prefill reselect mode=" in l and "picked=[" in l]
print("reselects (apply lines):", len(resel))
print("gpu scorer lines:", len(tot), "unavailable lines:", len(unav))
if tot:
    print("gpu total ms: median %.1f max %.1f min %.1f all %s" % (statistics.median(tot), max(tot), min(tot), tot))
same = setsame = 0
n = 0
worst = 1.0
for l in log:
    m = re.search(r"scorer shadow: ref=\[([\d,]*)\] fast=\[([\d,]*)\] (SAME|DIFFERENT) max\|dscore\| (\S+) \(rel (\S+)\) k-th/next block gap rel (\S+); ref (\S+) ms fast (\S+) ms", l)
    if not m:
        continue
    n += 1
    ref = m.group(1).split(","); fast = m.group(2).split(",")
    ov = len(set(ref) & set(fast)) / max(1, len(ref))
    worst = min(worst, ov)
    same += m.group(3) == "SAME"
    setsame += set(ref) == set(fast)
    print("shadow %d: %s ordered; set overlap %d/%d; max|d| %s rel %s gap %s cpu %s ms gpu %s ms" % (
        n, m.group(3), len(set(ref) & set(fast)), len(ref), m.group(4), m.group(5), m.group(6), m.group(7), m.group(8)))
print("shadow comparisons %d: identical ordered %d, identical as sets %d, worst overlap %.3f" % (n, same, setsame, worst))
