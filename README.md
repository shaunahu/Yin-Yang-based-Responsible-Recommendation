# Yin-Yang-based-Responsible-Recommendation
## File Structure
The file structure of this project is described as follows:
```
Yin-Yang-based-Responsible-Recommendation (rootpath)
| --- common
    |--- constants.py # file to save constants used globally
| --- model
    |--- message.py # defining Message class
    |--- user_agent.py # user agent model, defining through UserAgent class
    |--- recommender.py # Original recommendation system model, defining through Recommender class
| --- resource
| --- service
    |--- run.py # the entry of running a simulation
| --- tests # test folder for unit testing major functions
```

## Installation
1. Please make sure conda is installed.
2. To create conda environment, run
```
conda env create -f environment.yml
```

## data
1. `news_item_graph_full_augmented.pkl` includes the graph and edge data of 300 nodes subset `news_item_subset_300.csv`. 
2. Edge includes information: (1) topic similarity, (2) semantic similarity, (3) sentiment similarity

## Run
Executing simulator to train recommendation system with the following code:
```shell
python -m service.simulator
```
These files are automatically saved:
1. `recbole_results.log` saved in the project root folder, which is the evaluation and validation result from recbole model.
2. `saved/item_embedding.pkl`: item embedding from the embedding layer of the trained model.
3. `saved/user_embedding.pkl`: user embedding from the embedding layer of the trained model.
4. `saved/LighGCN.pth`: the trained model.