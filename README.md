# active-learning-for-Fault-Interpretation
This project is the code for paper URAL: Uncertainty-driven Region-based Active  Learning for Data-Efficient Fault Interpretation published in  Remote Sens
# Requirements
The main package and version of the python environment are as follows
| Name      | Version           |
|-------------------|---------------------|
|python             |       3.8.18        |           
|pytorch            |       2.2.0         |
|torchvision        |       0.17.0        | 
|matplotlib         |       3.7.5         |   
|opencv             |      4.9.0.80       |  
|pandas             |       2.0.3         |    
|scikit-learn       |       1.3.2         |     
|tqdm               |        4.66.2       |
|albumentations     |       1.4.18        |
|cmapy              |       0.6.6         |
|numpy              |        1.24.4       |


The above environment is successful when running the code of the project. Pytorch has very good compatibility. Thus, I suggest that try to use the existing pytorch environment firstly.

# Usage
## 1) Download Project
Running git clone https://github.com/Vinnie8609/active-learning-for-Fault-Interpretation.git

## 2) Datasets preparation
1.Download the datasets from the official address:
https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/YBYGBK

2.Modify the data folder path for specific dataset in data.py

## 3)Run Active learning process
Please confirm the configuration information in the [utils.py]
```bash
  python main.py \
      --seed 123 \
      --picknum 50 \
      --otherchoice transunet \
      --n_init_labeled 100 \
      --n_query 20 \
      --n_round 34 \
      --dataset_name THEBE \
      --strategy_name EntropySampling \


```
## 4)Pediction and Visualization
you can run model_predict_thebe to predict and visualize the whole process
