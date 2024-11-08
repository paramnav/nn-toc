

# Global Prediction Of Total Organic Carbon In Marine Sediments Using Deep Neural Networks (nn-toc)
<p align="center">
<img align="center" src="reports/figures/f03.png">
</p>

DOI: https://doi.org/10.3289/SW_3_2024

*Use of this codebase in research publication requires citation as follows:*

<code>
Parameswaran, Naveenkumar, Gonzales, Everardo, Burwicz-Galerne, Ewa , Braack, Malte and Wallmann, Klaus  (2024) Global Prediction Of Total Organic Carbon In Marine Sediments Using Deep Neural Networks (nn-toc).
DOI 10.3289/SW_3_2024
</code>

<br>
Here we create a deep neural network based approach for the geospatial predicition of total organic carbon percentages in marine sediments. For running the 
repostory, features and labels are dwonloaded from a data source. The data is then pre-processed using the notebooks and the scripts provided. In this, we 
provide the scripts to train the model, predict total organic percentages using Deep Neural Networks(DNNs), K Nearest Neighbours(KNN) and Random Forests. 
We compare the different methodologies based on the model performance. The uncertainty in the deep learning model is evaluated using Monte Carlo dropout. 
Information gain is used to quantify this uncertainty, which provides the expected knowledge gain from sampling at a certain location in the ocean. Below is
a folder structure or project organisation.  

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for the repository description
    ├── data
    │   ├── interim        <- Intermediate data that has been transformed. includes preprocessed features, selected features, masks for different marine regions, etc.
    │   ├── output         <- Final results from the model runs, that include the prediction maps, correlation plots, model performance, feature importance etc.
    │   └── raw            <- The original, immutable data, downloaded. Includes features and labels
    │    │
    ├── models             <- Trained deep learning models, or model summaries
    │
    ├── notebooks          
    │    └── TOC           <- /TOC has the notebooks to preprocess the data, run the models and postprocess the results.
    │         ├── MakeTrainingFeatures.ipynb
    │         ├── MakeGlobalFeatures.ipynb 
    │         ├── TOC_NN_CS.ipynb
    │         ├── TOC_NN_DO.ipynb
    │         ├── TOC_NN_entire.ipynb    
    │         ├── TOC_KNN_CS.ipynb
    │         ├── TOC_KNN_DO.ipynb
    │         ├── TOC_RF_CS.ipynb
    │         ├── TOC_RF_DO.ipynb
    │         ├── Visualisation.ipynb    
    │         ├── InfoGain.ipynb
    │         ├── ExtractLabels.ipynb   
    │         └── Infogain Experiment
    │                 ├── TOC_NN_CS_firsthalf.ipynb
    │                 ├── TOC_NN_CS_secondhalf.ipynb
    │                 ├── TOC_NN_DO_firsthalf.ipynb
    │                 ├── TOC_NN_DO_secondhalf.ipynb
    │                 ├── TOC_NN_DO_2_3.ipynb
    │                 └── TOC_NN_CS_2_3.ipynb
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    └── src                <- Source code for use in this project.
        ├── preprocessing  <- Scripts to preprocess features and labels
        │     ├── makeGlobalFeatures.py
        │     └── makeTrainingFeatures.py
        │
        ├── postprocessing <- Scripts to postprocess and visualise model results
        │     ├── getmodelPerformance.py
        │     ├── CurveFitting.py
        │     └── makeTrainingFeatures.py
        │
        └── models         <- Scripts to train DNN models and then use trained models to make predictions, and explain the trained model
              ├── trainDNN,py
              ├── predictDNN.py
              └── explainDNN.py              



--------
With these python files and notebooks, we provide instructions for reproducing the methods and results, that are described in detail in the [submitted paper (Parameswaran et. al. 2024 _GMD_)](https://www.egucopernicus.com/xxxxxxx).

* Start by cloning/forking the repo
``` 
git clone https://git.geomar.de/open-source/nn-toc.git
```

* Enter the local repository

```
cd nn-toc/
```

* Create a virtual environment using conda with the dependencies listed in the `nn-toc.yml` file

```
conda env create --file nn-toc.yml
```

* Activate the virtual environment

```
conda activate nn-toc
```



* The entire set of feeatures and labels used in the project can be accessed from https://zenodo.org/records/11186224 (https://doi.org/10.5281/zenodo.11186224). 


* Please download the data folder and paste it inside nn-toc, to run the models and make sure that the file structure of data is followed.




### A)  Make Training Features
In order to make pre-process features for training the different models, we use the nn-toc/notebooks/TOC/MakeTrainingFeatures.ipynb

This notebook performs feature selection, extracts selected features from measurement locations and saves and creates the feature-label dataset for the models. 

This notebook uses the functions from src/preprocessing/makeTrainingFeatures.py.



### B)  Make Global Features

For the global prediction, we need the features globally.

In order to make global features for prediction from different models, we use the nn-toc/notebooks/TOC/MakeGlobalFeatures.ipynb

In order to ease the memory requirements, we create chunks of global features.

This notebook uses functions from src/preprocessing/makeGlobalFeatures.py.


### C)  Running the different models.

Each method (Deep Neural Network: NN, K Nearest Neighbours: KNN, Random forests: RF) has a separate model for deep ocean and continental shelves. 

The naming is as follows: nn-toc/notebooks/TOC/TOC_methodName_MarineRegion.ipynb

The scripts loads the features, builds the model, trains the model, gets the model performance, gets the feature importance(only for DNN), and predicts 
the total organic carbon concentrations globally using the model.

The script for the DNN uses the functions from src/models/trainDNN.py, src/models/predictDNN.py, and src/models/explainDNN.py

Please note that some of the processes in this notebook are GPU intensive.

The section of script to compile the ensemble of predictions from the Monte Carlo dropout is memory intensive.

### D)  Information Gain 

With the compiled ensemble of prediction distirbutions, we can obtain the information gain map. For this run the script:

```
python3 src/postprocessing/CurveFitting.py
```

The script uses concurrent.futures for multi-processing and is computationally intensive.

After this, use the jupyter notebook nn-toc/notebooks/TOC/InfoGain.ipynb

This notebook evaluates the KL divergence (or Information gain) between the true distribution and the predicted distibution from the Monte Carlo dropout ensemble. 

To check if information gain works and actually brings in improvement in the model, we did an experiment, by comparing outputs from a model trained with points of more information gain with a model trained with points of low information gain. This experiment is included in /notebooks/TOC/Infogain\ Experiment/.



### E)  Visualisation

We visualise the results from differnt methods in nn-toc/notebooks/TOC/Visualisation.ipynb, which uses the functions from src/postprocessing/visualizePredictionMap.py

We calculate the TOC stock globally and in different marine regions and tabulate the results in the manuscript.
