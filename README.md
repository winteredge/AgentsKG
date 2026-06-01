# AgentsKG

AgentsKG is an automated knowledge graph generation framework based on Large Language Models (LLMs). By orchestrating multiple specialized agents to work collaboratively, the framework automates the generation, verification, alignment, and optimization of knowledge graphs.

## Project Structure

```
AgentsKG/
├── agentskg/       # Core package
│   ├── agents/     # Implementations of various agents
│   ├── core/       # Core data structures and types
│   ├── utils/      # Utility functions
│   └── pipeline/   # Pipeline orchestration
├── datasets/       # Experimental datasets (supports WebNLG and other formats)
├── evaluate/       # Evaluation scripts for experimental results
├── tests/          # Unit tests
├── environment.yml # Conda environment configuration file
└── README.md       # Project README file
```

## Quick Start

### 1. Environment Setup

```bash
# 1. Clone the repository
git clone https://github.com/winteredge/AgentsKG.git
cd AgentsKG

# 2. Create and activate the Conda environment using environment.yml
conda env create -f environment.yml
conda activate agentskg
```

### 2. Environment Variables

Please create a **`.env`** file manually in the root directory of the project and fill in your configurations:

```env
# Embedding Model Configuration
Embedding_MODEL = ""
EMBEDDING_API_URL = ""
EMBEDDING_API_KEY = ""

# OpenAI API Configuration
API_KEY = ""
API_URL = ""
MODEL = ""
```

## Running Experiments

To run the experiments, execute the corresponding Python scripts directly. Before running, please **modify the configurations or file paths directly in the code** according to your requirements.

### 1. Running the Main Knowledge Graph Generation Pipeline

Open `agentskg/pipeline/pipeline.py`. Navigate to the `if __name__ == "__main__":` entry point at the bottom of the file (or the configuration file) and modify the dataset you want to run:

```python
DATASET_TO_RUN = "aida-conll"  # Options: "webnlg", "CaRB", "aida-conll"
PROMPT_STRATEGY = "few_shot"   # Options: "zero_shot", "few_shot"
```

After modifying, run the script directly in your terminal:

```bash
python -m agentskg.pipeline.pipeline
```

### 2. Running Baseline Experiments

Open the corresponding baseline script in the `baseline/` directory, modify the data paths, and run the baseline:

```bash
# Example command (adjust script name as needed)
python baseline/run_baseline.py
```

### 3. Running Evaluation

Open `evaluate/evaluate_results.py`. Verify that the prediction file path and the ground truth path in the code are correct, and then run the evaluation:

```bash
python evaluate/evaluation_script.py
```

### 4. Data Format Conversion

If you need to convert the WebNLG dataset format, open `agentskg/utils/webnlg_convert_format.py` to modify the input and output directory paths, and then run:

```bash
python agentskg/utils/webnlg_convert_format.py
```
