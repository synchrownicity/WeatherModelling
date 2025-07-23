## **Forecasting the Unseen: Predicting Relative Humidity With Classical and Deep Learning Approaches**
_Team 2: Genie Tay Ting, Hui Yu Chao, Gan Zhi Yu Charlene, Yeo Jaye Lin_

## **Project Report**
Below is the link to our project report, detailing our key findings and figures related to this project. 

[IT1244 Project Report.pdf](https://github.com/user-attachments/files/21361057/IT1244.Project.Report.pdf)

## **Overview**
This project explores four machine learning techniques for **hourly forecasting of relative humidity** in Austrailia, using meteorological data collected near Monash University from **1 Jan 2010** to **31 May 2021**.

| Model Type     | Algorithms Implemented            | Key Notes                                       |
|----------------|-----------------------------------|-------------------------------------------------|
| Classical      | Multiple Linear Regression and Random Forest      | Baseline and ensemble benchmarks                |                                                
| Deep Learning  | Recurrent Neural Network (RNN) and Long Short-Term Memory (LSTM)   | Sequence-aware models for time-series trends    |                                          


Appropriate data engineering techniques and hyperparameter tuning were implemented where necessary to minimise errors and bring about accurate prediction. Predictions were visualised against ground truth in our test dataset to inspect trend fidelity.

### **Key Findings**
Our analysis revealed that the LSTM model delivered the highest accuracy, achieving the lowest RMSE and MAE, while most faithfully capturing diurnal and seasonal humidity patterns.

This highlights the value of sequence-aware deep learning for environmental forecasting, and provide a reproducible framework that can be extended to other climate variables or locations.

## **How To Run The Project** 
Our submission contains 2 main folders:
- **Code Folder** (contains our `IT1244_Team 2_Code File.ipynb` + other source code)
- **Model and Dataset Folder** (contains all of our models and `weather.csv` dataset)

Please first ensure you have the dependencies installed to the versions are listed in the `requirements.txt` file.

**To run our `.ipynb` code file, please take the following steps.**

If you are running our `.ipynb` file on Jupyter Notebook:
- Take **all** of the files present in the Model and Dataset folder, and place them in the same directory as the `ipynb` file.
- Execute the `.ipynb` in code block order (top to bottom) to load all of our models and view our plots.

If you are running our `.ipynb` file on Google Colab:
- Take **all** of the files present in the Model and Dataset folder, and place them into the _Files_ tab on the left sidebar.
- Take also the `data_processing.py` file present in our Code Folder, and also place it into the _Files_ tab on the left sidebar.
- Execute the `.ipynb` in code block order (top to bottom) to load all of our models and view our plots.

Please note that the `.ipynb` code file only contains the relevant code needed to visualise the various parameters and features of our model. If you would like to replicate the models on your own, please run the other source codes present in the Code Folder (those with a `.py` extension) to create the different models as labelled on your local repository. Note that a GPU is recommended when running the `.py` codes, especially for our deep learning models RNN and LSTM source code files.
