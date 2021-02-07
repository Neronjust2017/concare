# ConCare: Personalized Clinical Feature Embedding via Capturing the Healthcare Context

The source code for *ConCare: Personalized Clinical Feature Embedding via Capturing the Healthcare Context*

Our paper can be found [here](https://www.researchgate.net/publication/337481368_ConCare_Personalized_Clinical_Feature_Embedding_via_Capturing_the_Healthcare_Context). 
Thanks for your interest in our work.

## Visualization Tool
Welcome to try the prototype of our visualization tool (AdaCare):

http://47.93.42.104/215 (Cause of death: CVD)   
http://47.93.42.104/318 (Cause of death: GI disease)   
http://47.93.42.104/616 (Cause of death: Other)   
http://47.93.42.104/265 (Cause of death: GI disease)    
http://47.93.42.104/812 (Cause of death: Cachexia)   
http://47.93.42.104/455 (Cause of death: CVD)       
http://47.93.42.104/998 (Alive)       
http://47.93.42.104/544 (Alive)    

AdaCare can be found [here](https://github.com/Accountable-Machine-Intelligence/AdaCare), which is our another work in AAAI-2020.

Welcome to test the prototype of our visualization tool. The clinical hidden status is built by our latest representation learning model ConCare.
The internationalised multi-language support will be available soon.

## Requirements

* Install python, pytorch. We use Python 3.7.3, Pytorch 1.1.
* If you plan to use GPU computation, install CUDA

## Data preparation
We do not provide the MIMIC-III data itself. You must acquire the data yourself from https://mimic.physionet.org/. Specifically, download the CSVs. To run decompensation prediction task on MIMIC-III bechmark dataset, you should first build benchmark dataset according to https://github.com/YerevaNN/mimic3-benchmarks/.

After building the **in-hospital mortality** dataset, please save the files in ```in-hospital-mortality``` directory to ```data/``` directory.

## Run ConCare

All the hyper-parameters and steps are included in the `.ipynb` file, you can run it directly.
