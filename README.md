# DeepSeparator
### Experiment 1 
Choose best Encoder-Decoder
### Experiment 2 
Choose best Filter
### Experiment 6
Compare with traditional method
#### Prerequisite
```
pip install tftb
pip install pyhht
pip install nolds
pip install EMD-signal
```
#### Usage

1. Run tradition.py

Traditional method:

Change line 64 range(), range(0,400) for EOG, range(4000,4400) for EMG

Change sample number (400) accordingly.

Accordingly, change the text file name 

2. Run testCriterion.py

CNN method:

Change line 64 range(), range(0,400) for EOG, range(4000,4400) for EMG

Change sample number (400) accordingly.

Accordingly, change the text file name 

3. Run paint.py


#### Result sample
In result, the CNN result is derived from the complete data.
The traditional result is derived from 2 piece of data since I did not finish running.

