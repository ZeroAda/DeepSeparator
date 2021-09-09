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
Result in the process
#### Result
SNR picture and result in excel document.

### Log
2021/9/7 Update Experiment 6 test criterion from MSE to RRMSE
2021/9/9 Update Exp1, 2,4,6 test Criterion from MSE to RRMSE
