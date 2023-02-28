# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

results = np.array([(0, 0.0025), (0.001, 0.025), (0.005, 0.10), (0.01, 0.175), (0.05, 0.3), (0.1, 0.35)])
plt.plot(results[:, 0], results[:, 1])
plt.yscale('log')
plt.yticks([1, 1e-1, 1e-2, 1e-3, 1e-4])
plt.xlabel("Deletion Probability")
plt.ylabel("BER")
plt.title("BER of Conv autoencoder for Deletion AWGN channel\n m=100 r=1/2 SNR=6")
plt.savefig('../saves/conv Deletion AWGN.png', dpi=300)
plt.clf()

results = np.array([(-1, 0.105), (0, 0.080), (1, 0.0562), (2, 0.0358), (3, 0.0219), (4, 0.012), (6, 0.002), (8, 0.0002)])
plt.plot(results[:, 0], results[:, 1])
plt.yscale('log')
plt.yticks([1, 1e-1, 1e-2, 1e-3, 1e-4])
plt.xticks(results[:, 0])
plt.xlabel("SNR")
plt.ylabel("BER")
plt.title("BER of Conv autoencoder for AWGN channel\n m=100 r=1/2")
plt.savefig('../saves/conv AWGN.png', dpi=300)
