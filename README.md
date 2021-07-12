# jne-electrode-design
Release of source data, as well as simulation and analysis codes used in article "Do not waste your electrodes – Principles of optimal electrode geometry for spike sorting", published in the Journal of Neural Engineering, available at https://doi.org/10.1088/1741-2552/ac0f49.

# Source data
16 average spike templates are provided, recorded from mouse hippocampus CA1 region, pooled from one recording session each in two male mice (2-5 months old, vGLUT3-ires-Cre and SOM-ires-Cre on C57Bl/6J background).

The experiments were approved by the Ethical Committee for Animal Research at the Institute of Experimental Medicine, Hungarian Academy of Sciences, and conformed to Hungarian (1998/XXVIII Law on Animal Welfare) and European Communities Council Directive recommendations for the care and use of laboratory animals (2010/63/EU) (license number PE/EA/2552-6/2016).

The datafile, templates.csv is a 20x(16x8) matrix, where eight subsequent columns represent one template waveform over eight channels of a single probe shank.

Channel numbering and recording parameters are proivded in the article.

# Code

The use of the two main python scripts relevant in reproducing our results is briefly described as follows.

Sets of semi-artificial recordings can be generated using spike_generator.py, given a templates.csv file structured as our provided example.

A helper bash script, clustering.sh, is provided to automate running KlustaKwik on large batches of generated recordings.

Analysis of results can be performed by running sorting_quality.py. Results will be csv files, entries listing the number of acceptable quality clusters obtained in a given test recording.

Data generation, as well as analysis options can be configured in a human readable parameter file, to be placed in the working directory. A complete, self-documenting example, config.txt is provided in this release. The use of a dedicated configuration file helps create a traceable record of experimental settings.

In the current release, the code files, the configuration file, as well as the template file should be placed into the working directory, where the generated recording files are expected to be placed and analysed. The working directory could be trivially parameterised by users.

To help manage the required dependencies for KlustaKwik as well as our codebase we provide a Conda environment file, environment.yml.
