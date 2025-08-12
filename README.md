# Subtraction estimate

Foreground removal is essential task for DECIGO to observe the primordial SGWB. This repository offers the tools to estimate how larget the foregrounds (full, subtraction error, unresolvable) will be.

## Environment

I used `igwn-py310` environment.


## How to use

```
./get_merger_rate_density.py --outdir data/bbh --kind BBH
./get_overlap_function.py --outdir data/bbh --kind BBH
./get_omegagw.py --outdir data/bbh --kind BBH --snrthreshold 20 --ndet 12
```

For estimating BNS foreground, please replace BBH with BNS.