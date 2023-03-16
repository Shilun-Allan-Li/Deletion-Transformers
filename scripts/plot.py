# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import torch

SNRs = (-1, 0, 1, 2, 3, 4)
metrics = ('BER', 'RLD', 'BLER')
encDecNames = [('Conv', 'Cvt'), ('Conv', 'Conv'), ('Cvt', 'Cvt')]

SNR_results = []

ConvCvt_results_SNR = []
for enc, dec in encDecNames:
    result = []
    for SNR in SNRs:
        checkpoint = torch.load(f'../runs/{enc} encoder {dec} decoder AWGN 100 to 200 SNR {SNR} deletion 0/checkpoint.pt')
        result.append([checkpoint[m] for m in metrics])
    SNR_results.append(result)

SNR_results = np.array(SNR_results)

for i, metric in enumerate(metrics):
    for j, (enc, dec) in enumerate(encDecNames):
        plt.plot(SNRs, SNR_results[j, :, i], label=f'{enc}{dec}AE')
    plt.yscale('log')
    plt.xticks(SNRs)
    plt.xlabel("SNR")
    plt.ylabel(metric)
    plt.title(f"{metric} results for AWGN channel\n m=100 r=1/2")
    plt.legend()
    
    plt.savefig(f'../saves/{metric} vs SNR.png', dpi=300)   
    plt.clf()
    
    
    
deletion_probs = (0, 0.001, 0.005, 0.01, 0.05, 0.1)
deletion_results = []

for enc, dec in encDecNames:
    result = []
    for p in deletion_probs:
        checkpoint = torch.load(f'../runs/{enc} encoder {dec} decoder AWGN 100 to 200 SNR 6 deletion {p}/checkpoint.pt')
        result.append([checkpoint[m] for m in metrics])
    deletion_results.append(result)
deletion_results = np.array(deletion_results)

for i, metric in enumerate(metrics):
    for j, (enc, dec) in enumerate(encDecNames):
        plt.plot(deletion_probs, deletion_results[j, :, i], label=f'{enc}{dec}AE')
    plt.yscale('log')
    plt.xscale('log')
    plt.xticks(deletion_probs[1:])
    plt.xlabel("Deletion Probability")
    plt.ylabel(metric)
    plt.title(f"{metric} results for AWGN Deletion channel\n SNR=6 m=100 r=1/2")
    plt.legend()
    
    plt.savefig(f'../saves/{metric} vs DeletionProb.png', dpi=300)   
    plt.clf()



