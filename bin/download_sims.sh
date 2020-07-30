#!/usr/bin/env bash
cd `dirname "$(realpath $0)"`/..
mkdir sims
cd sims
echo "Downloading into $PWD"
curl -O http://mwhite.berkeley.edu/PUMAsim/HImesh_0512_z100.fits
curl -O http://mwhite.berkeley.edu/PUMAsim/HImesh_0512_z200.fits
curl -O http://mwhite.berkeley.edu/PUMAsim/Sky_0512_z100_060.0-30.0.fits
curl -O http://mwhite.berkeley.edu/PUMAsim/Sky_0512_z100_135.0+0.00.fits
curl -O http://mwhite.berkeley.edu/PUMAsim/Sky_0512_z200_060.0-30.0.fits
curl -O http://mwhite.berkeley.edu/PUMAsim/Sky_0512_z200_135.0+0.00.fits
curl -O http://mwhite.berkeley.edu/PUMAsim/GLEAM_EGC_v2_trim.fits


