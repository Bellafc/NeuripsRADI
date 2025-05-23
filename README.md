# RADI: LLMs as World Models for Robotic Action Decomposition and Imagination

This repository contains the official implementation code for the paper "RADI: LLMs as World Models for Robotic Action Decomposition and Imagination".

## Overview

RADI (Robotic Action Decomposition and Imagination) is a unified framework that leverages memory-augmented hierarchical decomposition and environmental imagination to simulate and verify action outcomes. When discrepancies are detected, self-reflective loops are triggered to iteratively re-optimize action plans without external supervision.

## Key Features

- **Multi-layer Decomposition**: Goal-level, task-level, and action-level three-layer decomposition strategy.
- **Environmental Imagination**: Uses LLMs as world models to predict action outcomes
- **Memory and Reflection**: Experience learning and error analysis through WorldModelMemory
- **Self-Correction**: Automatic plan correction mechanism based on verification results
- **Multi-model Support**: Supports ChatGPT-4o and open-source LLMs (Llama-2, etc.)

## Requirements

### Software Dependencies
- Python 3.10
- PyTorch >= 1.12.0
- transformers >= 4.21.0
- sentence-transformers
- openai >= 0.27.0 (for ChatGPT)

## Installation

### 1. Environment Setup

```bash
# Create and activate conda environment
conda create -n RADI python=3.10
conda activate RADI

# Clone repository
git clone https://github.com/anonymous/RADI.git
cd RADI

# Install dependencies
pip install -r requirement.txt
```

### 2. VirtualHome Environment Setup

We conduct experiments in VirtualHome. Download VirtualHome executable file (v2.2.5) from [Download VirtualHome here](https://1drv.ms/u/s!Am9fgKqXV2C2bB8WJWKb4-NABSg?e=8FJOUA) and unzip it to `RADI/RADI/virtualhome/`:

```bash
# Download and extract VirtualHome to the following directory structure
RADI/
└── RADI/
    └── virtualhome/
        └── simulation/
            └── unity_simulator/
                └── v2.2.5/
                    └── linux_exec.v2.2.5_beta.x86_64
```

### 3. LLM Model Setup

**For open-source LLMs**, download models to the `pretrain/` directory:

```bash
RADI/
└── pretrain/
    ├── llama-2-7b-chat-hf/
    ├── llama-2-13b-chat-hf/
    ├── Meta-Llama-3-8B-Instruct/
    ├── bloom-3b/
    ├── bloom-7b/
    ├── chatglm3-6b/
    └── chatglm3-6b-32k/
```

**For ChatGPT**, configure your API key in the `test_openai.py` file:

```python
# Edit RADI/RADI/task-planning/test_openai.py and replace with your API key
import openai
client = openai.OpenAI(api_key="your_openai_api_key_here")
```

Note: The framework uses the OpenAI client directly through the `test_openai.py` module rather than environment variables.

### 4. Dataset Preparation

We evaluate our method on four datasets, three from LID (In-Distribution, NovelScenes, NovelTasks) and one created by ourselves (LongTasks). Download the four datasets from [Download datasets here](https://drive.google.com/drive/folders/1oD2Ecmfc5h0VxJYOo5MnjpmdPEp8rVxm?usp=sharing) and unzip to `RADI/RADI/data/test_init_env/`:

```bash
# Dataset directory structure
RADI/
└── RADI/
    └── data/
        └── test_init_env/
            ├── InDistributation.p
            ├── NovelScenes.p
            ├── NovelTasks.p
            └── LongTasks.p
```

You can run `RADI/RADI/data/create_long.py` to create the LongTasks dataset. We remove some samples due to environment bugs from LID.

### LLM Models

We employ various LLMs of different scales as the backbone, including Llama-2-chat (7B, 13B), bloom (3B, 7B), ChatGLM3-6B, etc. For open-source LLMs, please download models to `RADI/pretrain/`:

- Llama-2-7b-chat-hf
- Llama-2-13b-chat-hf  
- Meta-Llama-3-8B-Instruct
- bloom-3b, bloom-7b
- chatglm3-6b, etc.

For closed-source models like ChatGPT, please configure your API key in the `test_openai.py` file by replacing the API key string.

## Instruction Tuning

### Usage
Go to `RADI/Instruction-Tuning/`. Run `run_bloom-3b.sh` or `run_bloom-7b.sh` for fine-tuning bloom. Run `run_chatglm.sh` for fine-tuning ChatGLM3-6B or ChatGLM3-6B-32K. Run `run_llama-7b.sh` or `run_llama-13b.sh` for fine-tuning Llama-2-chat or LongAlpaca. You can modify the parameters like "dataset", "train_batch_size", "accumulation_steps" to fit your own training.

## Quick Start

### Core Script Instructions

Go to `RADI/RADI/task-planning/`. Run `bash scripts/llm_eval.sh` to evaluate open-source LLMs for "RADI", "Embodied", "ReAct", "RADI-goal", and "RADI-task". Run `bash scripts/llm_eval_demo.sh` to evaluate open-source LLMs for "RADI-ft". Run `bash scripts/gpt_eval.sh` to evaluate closed-source LLMs for "RADI", "Embodied", "ReAct".

### 1. Evaluate RADI Framework (using ChatGPT-4o)

```bash
cd RADI/RADI/task-planning/

# First, configure your API key in test_openai.py
# Edit test_openai.py and replace "your_openai_api_key_here" with your actual key

# Evaluate on single dataset
python llm_eval.py \
    --llm gpt-4o \
    --mode multi-layer \
    --subset NovelTasks \
    --max_retry 3 \
    --base-port 8679
```

### 2. Using Open-source LLMs

```bash
# Using Llama-2 model
python llm_eval.py \
    --llm ../../pretrain/llama-2-13b-chat-hf \
    --mode multi-layer \
    --subset NovelTasks \
    --max_retry 1
```

## Core Module Description

### 1. World Model Verification (`world_model.py`)

```python
from world_model import verify_action_plan, WorldModelMemory

# Initialize memory manager
memory = WorldModelMemory(memory_file="worldmodel_memory.json")

# Verify action plan
world_model_str = "..."  # World model description
action_plan = "..."      # Action plan
verification_result, is_executable = verify_action_plan(
    world_model_str, action_plan, memory=memory
)
```

### 2. LLM Policy (`llm_policy.py`)

```python
from llm_policy import LLMPolicy

# Initialize LLM policy
llm_policy = LLMPolicy(args, logging)
llm_policy.reset(ids=task_id)
llm_policy.set_graph(env_graph)
llm_policy.set_goal(task_goal)

# Generate multi-layer decomposition plan
llm_policy.generate_multi_layer_plan()

# Get execution actions
while True:
    action = llm_policy.get_action_from_llm()
    if action == 'DONE':
        break
    # Execute action...
```

### 3. Interactive Evaluation (`interactive_evaluation.py`)

Contains complete RADI framework evaluation process:
- World model construction and enhancement
- Action plan verification and correction
- Multi-round reflection mechanism
- Performance metric calculation

## Configuration Parameters

### Key Parameters

* `base_port`: port number for VirtualHome environment
* `llm`: the path to LLM backbone (use "gpt-4o" for ChatGPT)
* `lora`: the path to lora weight, "None" for using LLM backbone only or closed-source LLMs
* `mode`: select task planning method: "multi-layer" for "RADI", "react" for "ReAct", "embodied" for "Embodied"
* `demo`: add this to use demonstrations
* `max_retry`: the number of times that task planning models can try, we set 1 in our experiment, you can set larger for higher success rate but longer inference time, it is useful for generating more training corpus

**Note**: API key configuration is handled in `test_openai.py`, not through command line parameters.

## Experiment Reproduction

### Table 2: Main Experimental Results

```bash
# Run RADI evaluation on all datasets (configure API key in test_openai.py first)
for subset in InDistributation NovelScenes NovelTasks LongTasks; do
    python llm_eval.py \
        --llm gpt-4o \
        --mode multi-layer \
        --subset $subset \
        --max_retry 99 \
        --base-port $((8679 + $RANDOM % 100))
done
```

## Output Results

### Evaluation Metrics

The program will output the following performance metrics:

```
**************** Current Evaluation Metric ****************
Successful / Executable / Current / Total: 45 / 52 / 60 / 100
Success Rate: 75.00
Executability: 86.67
```

### Result Files

- `success_rate_results.txt`: Success rate summary for each dataset
- `worldmodel_memory.json`: World model memory file
- `log.log`: Detailed execution logs

## Common Issues

### Q: VirtualHome connection failed
A: Check executable file path and port settings:
```bash
# Ensure path is correct
ls RADI/RADI/virtualhome/simulation/unity_simulator/v2.2.5/

# Try different port
python llm_eval.py --base-port 8680
```

### Q: OpenAI API call failed
A: Check API key configuration in `test_openai.py`:
```bash
# Edit test_openai.py and ensure your API key is correctly set
# Test API connection
python test_openai.py
```

### Q: Dataset files not found
A: Ensure dataset paths are correct:
```bash
# Check dataset files
ls RADI/RADI/data/test_init_env/
# Should contain: InDistributation.p, NovelScenes.p, NovelTasks.p, LongTasks.p
```

## File Structure

```
RADI/
├── RADI/RADI/task-planning/
│   ├── llm_eval.py              # Main evaluation script
│   ├── interactive_evaluation.py # RADI evaluation logic
│   ├── llm_policy.py           # LLM policy implementation
│   ├── world_model.py          # World model and verification
│   ├── arguments.py            # Parameter configuration
│   ├── init_path.py            # Path initialization
│   ├── sim_compute.py          # Similarity computation
├── RADI/RADI/data/             # Data directory
├── RADI/RADI/virtualhome/      # VirtualHome environment
└── pretrain/                   # Pre-trained model directory
```
