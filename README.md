This is a clone of <a href='https://github.com/WenjieDu/PyPOTS'>PyPOTS</a>.

The environment can be created using pip using requirements.txt file.

Please create a results directory under pypots directory.

There are five files named 'experiment_\*.py' under pypots/tests, where * stands for 05, 10, 15, 20 and 25.
These files can be run directly and the results will be saved under pypots/results folder as pickle files. Each pickle file contains results for Mean, Brits, Saits and Csaits for the feature masked.
Hence, for a given experiment_\*.py file, we expect 6 files to be created (one per individual feature masked and one for all features masked).
In each experiment_\*.py file, there is an absolute path mentioned - ensure that it points to the root folder. Since, it is an absolute path you may have to update it according to where the file is being run.
