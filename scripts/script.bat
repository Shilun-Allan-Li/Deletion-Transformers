for %%m in (0.001 0.005 0.01 0.05 0.1) do for %%n in (6) do python train.py --log_name "Cvt encoder Cvt decoder AWGN 100 to 200 SNR %%n deletion %%m" --SNR %%n --channel_prob %%m
