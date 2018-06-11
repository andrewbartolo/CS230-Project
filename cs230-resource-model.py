#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter
import pathlib
import sys, os

PLOT_DPI = 600 #1200
OUT_DIR = './cs230-out/'


# Defines the resource overhead model, per the pseudocode given in the poster + report.
#
K = 10 # We assume "adequate parallelism" such that the entire K attempts can be done in parallel by workers
# These were measured empirically from the MNIST code -
# see the start_time = time.time()... code in MNIST-NN.py.
FWD_PROP_RUNTIME = 0.0164
BACKPROP_RUNTIME = 0.0055
RANDSHIFT_RUNTIME = 0.0151
UPDATE_RUNTIME = 0.001196

# NOTE: I didn't have time to measure these empirically, but setting them to 0
# leads to a conservative savings estimate for H-ES
# (since BASE_TRANSMIT/BASE_RECEIVE/BASE_COMBINE will take longer than
# H_TRANSMIT/H_RECEIVE/H_COMBINE, as they're sending/processing more data over the network).
BASE_TRANSMIT_RUNTIME = 0
BASE_RECEIVE_RUNTIME = 0
BASE_COMBINE_RUNTIME = 0

H_TRANSMIT_RUNTIME = 0
H_RECEIVE_RUNTIME = 0
H_COMBINE_RUNTIME = 0


# Here, a "chunk" is 6 iterations (least common multiple of 1, 2, and 3, for Baseline, r=2, and r=3)
# Note that we are calculating wall-clock runtime, so work done in parallel is only counted once.
BASE_RUNTIME_PER_CHUNK = 6*(FWD_PROP_RUNTIME + BACKPROP_RUNTIME + UPDATE_RUNTIME
                            + BASE_TRANSMIT_RUNTIME + BASE_RECEIVE_RUNTIME + BASE_COMBINE_RUNTIME)

H_FULLGRAD_RUNTIME = (FWD_PROP_RUNTIME + BACKPROP_RUNTIME + UPDATE_RUNTIME
                      + H_TRANSMIT_RUNTIME + H_RECEIVE_RUNTIME + H_COMBINE_RUNTIME)

H_STOCH_RUNTIME = (FWD_PROP_RUNTIME + RANDSHIFT_RUNTIME + UPDATE_RUNTIME
                   + H_TRANSMIT_RUNTIME + H_RECEIVE_RUNTIME + H_COMBINE_RUNTIME)


R2_RUNTIME_PER_CHUNK = 3*H_FULLGRAD_RUNTIME + 3*H_STOCH_RUNTIME
R3_RUNTIME_PER_CHUNK = 2*H_FULLGRAD_RUNTIME + 4*H_STOCH_RUNTIME

print(BASE_RUNTIME_PER_CHUNK)
print(R2_RUNTIME_PER_CHUNK)
print(R3_RUNTIME_PER_CHUNK)

# Percentage for table values.
print(float(R2_RUNTIME_PER_CHUNK)/BASE_RUNTIME_PER_CHUNK)
print(float(R3_RUNTIME_PER_CHUNK)/BASE_RUNTIME_PER_CHUNK)
print('-'*40)


# Memory consumption section
# These are estimated from the live variables in the MNIST-NN Python code
# Assume 1-byte elements, but it doesn't matter (word size cancels out)
# W1, dW1 dims: 300x784
# W2, dW2 dims: 10x300
# b1, db1 dims: 300x1
# b2, db2 dims: 10x1
W1_M = 300*784
W2_M = 10*300
b1_M = 300
b2_M = 10
z1_M = 300
a1_M = 300
z2_M = 10
a2_M = 10

# Forward prop
FWD_PROP_MEM = W1_M + W2_M + b1_M + b2_M + z1_M + a1_M + z2_M + a2_M
# 2* to account for both parameters and their gradients
BACKPROP_MEM = 2*(W1_M + W2_M + b1_M + b2_M) + a2_M + z2_M + a1_M + z1_M
# Update mem can use already-cached data (so it's 0)

# Again, assume these 0 as I didn't have time to measure them empirically,
# (Would need to take into account buffering, PRNG memory consumption, etc.),
# but they should be fairly similar between Baseline and H-ES
TRANSMIT_M = 0
RECEIVE_M = 0
COMBINE_M = 0


BASE_MEM_PER_CHUNK = 6*(FWD_PROP_MEM + BACKPROP_MEM + TRANSMIT_M + RECEIVE_M + COMBINE_M)
R2_MEM_PER_CHUNK = 3*(FWD_PROP_MEM + BACKPROP_MEM) + 3*(FWD_PROP_MEM)
R3_MEM_PER_CHUNK = 2*(FWD_PROP_MEM + BACKPROP_MEM) + 4*(FWD_PROP_MEM)

print(BASE_MEM_PER_CHUNK)
print(R2_MEM_PER_CHUNK)
print(R3_MEM_PER_CHUNK)

# Percentage for table values.
print(float(R2_MEM_PER_CHUNK)/BASE_MEM_PER_CHUNK)
print(float(R3_MEM_PER_CHUNK)/BASE_MEM_PER_CHUNK)
print('-'*40)


# Network traffic section
# Note that we model TX and RX together, as any byte sent has to be received somewhere within the network (and we don't care about external traffic)
#
FULL_BW = W1_M + W2_M + b1_M + b2_M
H_BW = 2 # random seed + best cost

BASE_BW = 6*FULL_BW
R2_BW = 3*FULL_BW + 3*H_BW
R3_BW = 2*FULL_BW + 4*H_BW

print(BASE_BW)
print(R2_BW)
print(R3_BW)

# Percentage for table values.
print(float(R2_BW)/BASE_BW)
print(float(R3_BW)/BASE_BW)
print('-'*40)


# For generating the plots
N = 3
baselineOverheads = (float(BASE_RUNTIME_PER_CHUNK)/BASE_RUNTIME_PER_CHUNK, float(BASE_MEM_PER_CHUNK)/BASE_MEM_PER_CHUNK, float(BASE_BW)/BASE_BW)

fig, ax = plt.subplots()

ind = np.arange(N)    # the x locations for the groups
width = 0.20         # the width of the bars
opacity = 0.85
p1 = ax.bar(ind, baselineOverheads, width, color='#7e0308', alpha=opacity, bottom=0)


r2Overheads = (float(R2_RUNTIME_PER_CHUNK)/BASE_RUNTIME_PER_CHUNK, float(R2_MEM_PER_CHUNK)/BASE_MEM_PER_CHUNK, float(R2_BW)/BASE_BW)
p2 = ax.bar(ind + width, r2Overheads, width, color='#396190', alpha=opacity, bottom=0)

r3Overheads = (float(R3_RUNTIME_PER_CHUNK)/BASE_RUNTIME_PER_CHUNK, float(R3_MEM_PER_CHUNK)/BASE_MEM_PER_CHUNK, float(R3_BW)/BASE_BW)
p3 = ax.bar(ind + 2*width, r3Overheads, width, color='gray', alpha=opacity, bottom=0)

ax.set_title('Overheads normalized to parallel BGD')
ax.set_xticks(ind + width)# / 2)
ax.set_xticklabels(('Runtime', 'Memory', 'Network BW'))

ax.legend((p1[0], p2[0], p3[0]), ('Baseline', 'H-ES, r = 2', 'H-ES, r = 3'))
ax.autoscale_view()

if len(sys.argv) > 1:
    # ensure that the out dir exists
    pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    # export as EPS, PNG, and SVG
    fig.savefig(OUT_DIR+'plot.eps', format='eps', dpi=PLOT_DPI)
    fig.savefig(OUT_DIR+'plot.png', format='png', dpi=PLOT_DPI)
    fig.savefig(OUT_DIR+'plot.svg', format='svg', dpi=PLOT_DPI)

plt.show()
