#!/bin/sh
# Move the tandem-c and tandem-cuda pins to the latest upstream main.
git -C "$(dirname "$0")/.." submodule update --remote external/tandem-c external/tandem-cuda
