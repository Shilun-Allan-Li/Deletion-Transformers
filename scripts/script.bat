for %%n in (-1 0 1 2 3 4) do python train.py --log_name "AWGN 100 to 200 SNR %%n Conv encoder Conv decoder" --SNR %%n
