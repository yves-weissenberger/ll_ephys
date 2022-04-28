# Overview

There is code and data. Data is in folders which corresponds to individual sessions from individual subjects. Each session is comprised of a task session followed by an open field session. The naming conventions for the folder containing the data is year-month-day_subjectID_kilosortversion.
The easiest way to understand the meaning of the files is to look at ./code/scripts and open either the .py or the .ipynb files. The files are explained where they are loaded for the example analyses

# Instructions


## Python
Go to ./code/scripts to find jupyter notebooks which explain/show how to use the data. For people who don't use jupyter notebooks there are also .py files in there which do the exact same thing. There should also be a requirements.txt in ./code


## Matlab

The npy-matlab-master package allows you to import .npy files (which are arrays in python) into matlab. All of the data is either in this format or in .csv format. Look at load_data_test.m to see how to load it. To see how to use it look first at the .py files in ./code/scripts and then at files in ./code/mecll, which can all just be opened with a text editor. Shouldn't be too hard to turn this into matlab.