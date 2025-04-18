# AIHSG
generate
This project aims to implicit generate hate speech examples for analysis and research. The following is a detailed guide on how to set up the environment, install dependencies, preprocess data, and run experiments.

# 1. Create a Virtual Environment
It is recommended to use a virtual environment to manage the project's dependencies and avoid conflicts with the system's global environment. You can use venv or conda to create a virtual environment.
```bash  
conda create -n myenv python=3.8
conda activate myenv
```
# 2. Dependency Installation
After activating the virtual environment, install the project's dependencies. You can use pip to install all the dependencies listed in the requirements.txt file. First, make sure you have created the requirements.txt file with the following content:
```plaintext 
torch
requests
nltk
sentence-transformers
transformers
omegaconf
hydra-core
pandas
mlflow
openai
```
# 3.Configuration File Explanation
The project's configuration information is stored in the AIHSG/experiments/conf/config.yaml file. 

# 4.Data Preprocessing
Before running the experiment, ensure that the training data and lexicon data are in the correct format. 

# 5.Running the Experiment
After completing the environment setup and dependency installation, you can run the experiment to generate hate speech examples. During the experiment, hate speech examples will be generated and saved as a CSV file. At the same time, the experiment's configuration information and generated data will be logged to MLflow for easy subsequent analysis and management.

# 6.Experiment Results
After the experiment is completed, the generated hate speech examples will be saved as a gpt3-{engine}__shots={nof_shots}.csv file, where {engine} is the generation model endpoint specified in the configuration file, and {nof_shots} is the number of examples used as a prompt. Additionally, the experiment's configuration information and generated data will be logged to MLflow. You can access the detailed information of the experiment by visiting the uri_path specified in the configuration file.

# 7.Notes
  Make sure you have correctly configured all the parameters in the AIHSG/experiments/conf/config.yaml file before running the experiment, especially the secret_key.
  The generation of hate speech examples is only for research and analysis purposes. Please comply with relevant laws, regulations, and ethical guidelines.
  Running the experiment may require a certain amount of computing resources and time. Please adjust accordingly based on your actual situation.
