# Subtraction estimate

Foreground removal is essential task for DECIGO to observe the primordial SGWB. This repository offers the tools to estimate how larget the foregrounds (full, subtraction error, unresolvable) will be.

Preprint [https://arxiv.org/abs/2601.18378]

## Environment

I used `igwn-py310` environment.


## How to use

First, you calculate the merger rate density as a function of redshift. It is carried out by the script `get_merger_rate_density.py`. This script will calculate the merger rate density that is normalized so that the local merger rate density is unity: $R_0 = 1 \mathrm{Gpc^{-3}\ yr^{-1}}$. The sample command is,

```
./get_merger_rate_density.py --outdir data/bbh --kind BBH
```
This command will create the directory `--outdir` and the dat file of merger rate density in that directory.


Second, you calculate the overlap function, which is the number of binaries located within one frequency bin. (Eq. (2.25) of [paper](https://arxiv.org/abs/2601.18378)). An example command is
```
./get_overlap_function.py --kind bbh --outdir data/bbh --merger_rate_file data/bbh/normalized_merger_rate_density_bbh.dat --local_merger_rate_density 20 --nsample 1000
```

Finally, you estimate the foreground. The sample code is
```
./get_omegagw.py --outdir data/251028_8channel_2corr_sypsd/gwtc4_simpleuniformbns --kind bns --merger_rate_file data/bns/normalized_merger_rate_density_BNS.dat --local_merger_rate_density 100 --nsample 1000 --snrthreshold 15
```
This will create a dat file and two png files. The dat file contains `frequency, Unresolvable, Separable, Subthreshold, Err, Err(projected), and Full` components of foreground. Png files are figures of the foregrounds.

<!-- 

BNS foreground also contributes to the binary detection and subtraction. To do this, we separate the calculations of full Omega_gw of BNS foreground and of each components of the foreground.

```
./get_full_omegagw.py --outdir data/260205bns_lvko4a --kind bns --merger_rate_file data/bns/normalized_merger_rate_density_BNS.dat --local_merger_rate_density 100 --nsample 1000
``` -->
