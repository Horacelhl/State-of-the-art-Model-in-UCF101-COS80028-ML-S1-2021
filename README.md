# State-of-the-art-Model-in-UCF101-COS80028-ML-S1-2021

Setting and environment Both Windows and Linux: 
  

Conda environment with Python 3.9:  
pandas  
OpenCV (Windows)  
tensorflow (Windows)  
tensorflow-gpu (OzStar)  
tensorflow hub 
sklearn 
matplotlib 
  
OzStar environment:  
Source activate conda environment  
module load anaconda3/5.1.0  

GPU modules:  
module load cudnn/8.1.0-cuda-11.2.0  
module load cuda/11.2.0 
  
SMART method: 
Data_processing.py 
SingleSelector.py 
GlobalSelector.py 
New-SingleSelector.ipynb 
New-GlobalSelector.ipynb 
New-SelectorModel.ipynb 
 
CNNs: 

Preprocessing_UCF101.py 
TimeDistributed_ResNet50_MLP.py 
TimeDistributed_ResNet50_LSTM_MLP.py 

Backend model: 
ResNet50+MLP after Selector model.ipynb 

 
Process pipeline: 

We need to perform “New-SingleSelector.ipynb” and “New-GlobalSelector.ipynb” training and get the trained weight. After that, we use “New-SelectorModel.ipynb” to complete the SMART method. After that, we use the SMART method output into “Preprocessing_UCF101.py”, and it will create a new folder. Finally, use “ResNet50+MLP after Selector model” to get the final results. 

All processes are based on Python only. 
