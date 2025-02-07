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