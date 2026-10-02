#!/bin/sh
# Copy the reference sources from tandem-c and tandem-cuda checkouts.
# Usage: tools/sync.sh path/to/tandem-c path/to/tandem-cuda
set -e
c=${1:-../tandem-c}
cuda=${2:-../tandem-cuda}
pkg="$(dirname "$0")/../src/tandem_torch"
cp "$c/tandem.c" "$c/tandem.h" "$pkg/c/"
cp "$cuda/tandem.cuh" "$pkg/cuda/"
