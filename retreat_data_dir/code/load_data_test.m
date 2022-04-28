%load .npy files from matlab
addpath('/path/to/npy-matlab-master')

%load the neural responses to each poke
task_firing_rate_maps = readNPY('/path/to/data/2021-08-06_39964_ks25/neuron_response_table.npy');


%load table describing what happened during each of the pokes
tab = readtable('/path/to/data/2021-08-06_39964_ks25/task_event_table.csv');
