# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import torch
import autoencoders
import channels
from argparse import Namespace

device = torch.device("cuda")

def plot_SNR():
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
    
    
def plot_deletion():
    deletion_probs = (0, 0.001, 0.005, 0.01, 0.05, 0.1)
    metrics = ('BER', 'RLD', 'BLER')
    encDecNames = [('Conv', 'Cvt'), ('Conv', 'Conv'), ('Cvt', 'Cvt')]
    
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


def calculate_power_error_position(model_name, SNR, deletion_prob):
    checkpoint = torch.load('../runs/{} encoder {} decoder AWGN 100 to 200 SNR {} deletion {}/checkpoint.pt'.format(*model_name, SNR, deletion_prob))
    args = Namespace(**checkpoint['args'])
    channel = lambda x: channels.binaryDeletionChannel(channels.AWGN(x, args.SNR)[0], args.channel_prob)
    AE_model = getattr(autoencoders, "{}{}AE".format(*model_name))(args, channel).to(device)
    AE_model.encoder.load_state_dict(checkpoint['encoder_state'])
    AE_model.decoder.load_state_dict(checkpoint['decoder_state'])
    
    AE_model.eval()
    
    position_errors = []
    powers = []
    
    with torch.no_grad():
        for step in range(10000 // args.batch_size):
            message = torch.randint(0, args.alphabet_size, (args.batch_size, args.message_length), device=device)
            
            codeword = AE_model.codeword(message)
            output = AE_model.predict(message).contiguous()
     
            predictions = output.argmax(-1)
            
            position_error = torch.mean((predictions != message).float(), dim=0)
            power = torch.mean(codeword**2, dim=0)
            
            position_errors.append(position_error)
            powers.append(power)
            
    position_errors = torch.stack(position_errors)
    powers = torch.stack(powers)
    
    position_error = torch.mean(position_errors, dim=0).cpu()
    power = torch.mean(powers, dim=0).cpu()
    
    return position_error, power

def plot_position_error():
    encDecNames = [('Conv', 'Cvt'), ('Conv', 'Conv'), ('Cvt', 'Cvt')]
    SNR = 6
    deletion_prob=0.005
    position_results = []
    power_results = []
    for model_name in encDecNames:
        position_error, power = calculate_power_error_position(model_name, SNR=SNR, deletion_prob=deletion_prob)
        position_results.append(position_error)
        power_results.append(power)
        
    for i, model_name in enumerate(encDecNames):
        plt.plot(range(100), position_results[i], label="{}{}".format(*model_name))
    plt.xlabel("Message bit positions")
    plt.ylabel("Error rate")
    plt.title(f"Bit Error Rate of Each Position\n SNR={SNR}, Deletion Prob={deletion_prob}")
    plt.legend()
    
    plt.savefig(f'../saves/BER vs Position (SNR_{SNR} p_{deletion_prob}).png', dpi=300)
    plt.clf()
    
    for i, model_name in enumerate(encDecNames):
        plt.plot(range(200), power_results[i], label="{}{}".format(*model_name))
    plt.xlabel("Codeword bit positions")
    plt.ylabel("Power")
    plt.title(f"Mean Power of Each Codeword Position\n SNR={SNR}, Deletion Prob={deletion_prob}")
    plt.legend()
    
    plt.savefig(f'../saves/Power vs Position (SNR_{SNR} p_{deletion_prob}).png', dpi=300)
    plt.clf()

plot_position_error()

