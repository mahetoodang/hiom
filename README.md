# Hierarchical Ising opinion model (HIOM)

## Requirements
* python 3.6
* virtualenv for python

## Setup
* clone repo
* create virtualenv: "python3 -m venv <env_name>"
* activate virtualenv: "source <env_name>/bin/activate" (for Mac)
* install requirements: "pip install -r requirements.txt"

## Using the model

### Reproducing the results
Run main.py. Model parameters can be changed, look src/Model.py for reference. Some plotting functions are called out in main.py, others are commented out or not imported. Look src/plotter.py for all current plotting possibilities

### Mesa visualization
In "visualization" folder run server.py. A small webapp provided by mesa package should open in browser. This is not meant for running actual experiments, but can be useful in order to familiarize with network topologies and look for example how rapidly attention and polarization increase among the agents.
