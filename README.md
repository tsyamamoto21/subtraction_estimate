# Subtraction estimate

Foreground removal is essential task for DECIGO to observe the primordial SGWB. This repository offers the tools to estimate how larget the foregrounds (full, subtraction error, unresolvable) will be.

## Environment

I used `igwn-py310` environment.


## How to use

```
./get_merger_rate_density.py --outdir data/bbh --kind BBH
./get_overlap_function.py --outdir data/bbh --kind BBH
```


Revised 2025.11.25

```
./get_omegagw.py --outdir data/251028_8channel_2corr_sypsd/gwtc4_simpleuniformbns --kind bns --merger_rate_file data/bns/normalized_merger_rate_density_BNS.dat --local_merger_rate_density 100 --nsample 1000 --snrthreshold 15
```