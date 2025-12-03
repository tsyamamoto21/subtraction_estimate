#!/bin/bash
for snrth in 40 60 80 100
do
    ./get_omegagw.py\
    --outdir data/251202bns_lvko4a/\
    --kind bns\
    --merger_rate_file data/251202bns_lvko4a/normalized_merger_rate_density_bns.dat\
    --local_merger_rate_density 100\
    --nsample 1000\
    --snrthreshold $snrth
done
