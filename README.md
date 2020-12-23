## Introduction
This is a backdoor detector for BadNets trained on the YouTube Face dataset.  
BadNets: all BadNets are under **models/** directory  
Repaired Models: under **entropy/** and **finePruning/** directory
Methods: STRIP + Fine-pruning
## Environment Requirements
 - python 3.x
 - os
 - numpy
 - keras
 - tensorflow
 - h5py
 - scipy
 - matplotlib
 - cv2 (opencv-python)
## Quick Start
1. Download the data from [here](https://drive.google.com/drive/folders/1FhMDxD4cezVNk7BhRVSbhdkRwXUTI7oK) into the **data/** directory.
2. Open your terminal and run:  
    > *python eval.py*
3. Follow the instructions from your terminal

## **Attention!!!**
***In the pruning process,  the running time may be very long. (e.g. for 10k data, may need 15min.)***  
***If you notice the following warnings from your terminal, please ignore them.***  
![](resources/warning.png)