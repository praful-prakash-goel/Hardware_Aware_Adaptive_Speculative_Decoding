# Hardware-Aware Adaptive Speculative Decoding

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/praful-goel/speculative_decoding_models)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)

## Problem Statement: The Limits of Static Speculation

In standard speculative decoding (Leviathan et al., 2023), the speculation lookahead depth ($\gamma$) is fixed as a static hyperparameter across the entire generation sequence (e.g., $\gamma = 5$). 

This static design creates a severe performance inefficiency due to the non-stationary entropy of autoregressive generation:
1. **Under-Speculation in Low-Entropy Regimes:** During highly predictable sequences (e.g., code syntax, common phrases, structured formats), the draft model aligns closely with the target model. A static $\gamma$ caps generation prematurely, leaving potential throughput gains unexploited.
2. **Compute Waste in High-Entropy Regimes:** During open-ended reasoning or creative text, prediction entropy spikes. If the target model rejects the draft model's prediction at position $i \ll \gamma$, all subsequent speculative tokens ($i+1 \dots \gamma$) are unconditionally discarded. The time, memory bandwidth, and kernel launches spent drafting those discarded tokens represent purely wasted overhead.

### The Breakdown of Model-Side Adaptive Methods
To address this inefficiency, prior literature proposed **Adaptive Speculative Decoding**, dynamically modulating $\gamma$ based on **model-side statistical proxies** (such as draft output entropy, top-1 softmax confidence, or rolling acceptance rates). 

However, model-side proxies suffer from a critical blind spot: **they assume computational cost is proportional to statistical alignment.** In resource-constrained and consumer hardware environments:
* Host-side CPU dispatch latency, kernel serialization, and memory bandwidth contention dominate single-token forward passes.
* A high token acceptance rate (e.g., $>85\%$) can still produce negative speedups ($<1.0\times$) if the physical latency of sequential draft launches exceeds the verification savings.
* Model-side signals are completely blind to physical hardware bottlenecks such as cache memory pressure, PCIe bus contention, and runtime kernel overhead.

---

## Solution: Hardware-Aware Adaptive Speculative Decoding

Rather than relying on model-derived heuristics that fail to capture physical execution costs, we introduce a **Hardware-Aware Adaptive Speculative Controller** driven directly by live wall-clock timing signals. 

By measuring real-time Tokens Per Second (TPS) via non-blocking hardware event timers and smoothing variance through an Exponential Moving Average (EMA), our runtime controller:
1. **Tracks Hardware-Optimal $\gamma$ Dynamically:** Adjusts speculation depth $\gamma$ to operate at the empirical peak of throughput, consistently halting speculation before cumulative drafting latency crosses the target verification threshold.
2. **Dynamically Manages KV Cache Strategy:** Co-optimizes cache allocation and rollback policies to avoid VRAM fragmentation and PCIe shared-memory thrashing on memory-constrained GPUs.
3. **Guarantees Zero Baseline Regression:** Automatically falls back to standard autoregressive decoding ($\gamma = 0$) whenever physical conditions (such as CPU dispatch bottlenecks or low acceptance) indicate that speculation is operating at a net loss, ensuring performance never drops below the baseline model.

### Key Objectives

1. **Granular Per-Stage Latency Profiling:** Dissect the end-to-end speculative execution loop into five isolated stages:
   * **Draft Speculation**
   * **Target Verification**
   * **Comparison & Rejection Sampling Logic**
   * **KV Cache Rollback**
   * **Tensor Slicing & State Updates**
2. **Empirical Bottleneck Identification:** Pinpoint the architectural factors that cause speculative decoding to collapse below baseline autoregressive speeds on consumer GPUs.
3. **Live Hardware-Adaptive Controller:** Implement an exponential moving average (EMA) feedback controller that dynamically steps $\gamma$ and toggles KV-caching strategies online, providing a strict performance guarantee that runtime throughput will not drop below the standalone autoregressive baseline.

---

## Key Research Findings

Experiments conducted on consumer GPU hardware (**NVIDIA RTX 3050 Ti Laptop GPU, 4GB VRAM**) across multiple model families yielded key systems-level insights:

* **Disproving the Cache Rollback Bottleneck Hypothesis:** Cache state rollback and tensor cropping account for **less than 0.1%** of total step execution time, refuting the assumption that cache manipulation causes speculative engine stalls.
* **Draft-to-Target TPS Ratio Dominance:** Speculative viability is fundamentally governed by the standalone draft-to-target execution ratio rather than the acceptance rate alone:
  * **SmolLM (135M / 1.7B):** High standalone draft throughput yields peak speedups up to **$3.35\times$** at $\gamma=5$.
  * **Pythia (160M / 1B):** Moderate ratio achieves **$1.75\times$** peak speedup at $\gamma=7$.
  * **SmolLM2 (135M / 1.7B):** Fails to surpass baseline autoregressive throughput ($<1.0\times$), despite maintaining high token acceptance rates ($>85\%$), caused by CPU host-dispatch serialization on small kernel launches.
* **The Draft/Verify Crossover Boundary:** As $\gamma$ increases, cumulative draft latency scales linearly. The optimal $\gamma$ is highly sensitive to the draft-verify execution crossover point and fluctuates dynamically based on hardware thresholds—a systems-level constraint that acceptance-rate-only controllers completely fail to detect.

---

## Model Architecture & Training

Evaluations encompass standardized open-source model pairs alongside custom transformer architectures trained from scratch:

### Standardized Model Suites

| Model Family | Target Model | Draft Model | Precision | Vocabulary |
| :--- | :--- | :--- | :--- | :--- |
| **Pythia** | Pythia-1B | Pythia-160M | FP16/BF16 | 50,304 |
| **SmolLM** | SmolLM-1.7B | SmolLM-135M | FP16/BF16 | 49,152 |
| **SmolLM2** | SmolLM2-1.7B | SmolLM2-135M | FP16/BF16 | 49,152 |

### Custom Models (Trained from Scratch)

Both custom models are decoder-only Transformers trained on OpenWebText using identical tokenizers to enforce strict statistical alignment without divergent token boundaries:

| Hyperparameters | Main Model (Target) | Draft Model (Small) | Draft Model (Medium) |
| :--- | :--- | :--- | :--- |
| **Parameters** | ~150M | ~30M | ~70M |
| **Layers** | 12 | 2 | 6 |
| **Heads** | 12 | 4 | 8 |
| **Embedding Dim** | 768 | 256 | 512 |
| **Context Length** | 1024 | 1024 | 1024 |
| **Vocab Size** | 50304 | 50304 | 50304 |
| **Droupout** | 0.1 | 0.1 | 0.1 | 
| **Dataset** | OpenWebText (Sample) | OpenWebText (Sample) | OpenWebText (Sample) |

## Download Pre-trained Models

You can download the trained weights directly from Hugging Face:

| Model | Parameters | Description | Link |
| :--- | :--- | :--- | :--- |
| **Main Model** | ~150M | The main larger target model which is used for the final output | [Download .pt](https://huggingface.co/praful-goel/speculative_decoding_models/resolve/main/main_model.pt) |
| **Draft Model (Small)** | ~30M | The smaller, lightweight version of the main model which is used to quickly predict the upcoming tokens | [Download .pt](https://huggingface.co/praful-goel/speculative_decoding_models/resolve/main/draft_small_model.pt) |
| **Draft Model (Medium)** | ~70M | The medium version of the main model which is used to quickly predict the upcoming tokens | [Download .pt](https://huggingface.co/praful-goel/speculative_decoding_models/resolve/main/draft_medium_model.pt) |

**Setup:**
1. Download both `.pt` files and their corresponding `.json` configs.
2. Place them in the `saved_models/` directory.

## Project Structure

```bash
Speculative_Decoding_Inference_Engine/
│
├── data/
│   ├── __init__.py
│   ├── data_loader.py
│   └── prepare_data.py
│
├── experiments/
│   ├── plots/
│   │   ├── baseline_tps.pdf
│   │   ├── ratio_vs_speedup.pdf
│   │   ├── smollm_profile.pdf
│   │   ├── speedup_gamma.pdf
│   │   └── stress_test.pdf
│   ├── results/
│   │   ├── benchmarks.csv
│   │   ├── benchmarks_pythia.csv
│   │   ├── benchmarks_smol.csv
│   │   ├── benchmarks_smol2.csv
│   │   ├── pythia_profile.csv
│   │   ├── smollm_profile.csv
│   │   ├── stess_test.csv
│   │   ├── stress_test_pythia.csv
│   │   ├── stress_test_smol.csv
│   │   └── stress_test_smol2.csv
│   ├── __init__.py
│   ├── benchmark_tps.py
│   ├── evaluate_alignment.py
│   ├── plot_graphs.py
│   └── profiler.py
│
├── inference/
│   ├── __init__.py
│   ├── generate.py
│   └── speculative_engine.py
│
├── model/
│   ├── __init__.pt
│   ├── config.py
│   └── model_architecture.py
│
├── saved_models/
│   ├── draft_medium_config.json
│   ├── draft_small_config.json
│   └── main_config.json
│
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
└── train.py
```

# Usage

## Installation

Because this project focuses heavily on systems benchmarking and runtime analysis, the primary entry points are the profiling and benchmarking scripts. Training custom models is fully supported but treated as a secondary feature for architectural ablation.
```bash
git clone https://github.com/praful-goel/Hardware-Aware-Adaptive-Speculative-Decoding.git
cd Hardware-Aware-Adaptive-Speculative-Decoding
pip install -r requirements.txt
```

---

## 1. Per-Stage Latency Profiling

Profile execution latency across the 5 internal speculative stages using synchronized CUDA events. This script generates the exact millisecond breakdowns of draft generation, target verification, and cache manipulation.

```bash
python -m experiments.profiler --model SmolLM --max_new_tokens 256
```

`profiler.py` arguments:

| Argument | Type | Default | Constraints | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--model` | str | `SmolLM` | one of {`custom`, `pythia`, `SmolLM`, `SmolLM2`} | Model family to profile. |
| `--max_new_tokens`| int | 256 | > 0 | Total sequence tokens to profile. |

The script will:
- Instrument the 5 stages of the speculative step.
- Measure average ms and percentage of total time per step.
- Output step-level timing distributions directly into `experiments/results/{model}_profile.csv`.

---

## 2. Benchmark Tokens Per Second & Stress Testing

Execute parameter sweeps and context stress tests across cached and non-cached configurations to evaluate throughput and speedup dynamics.

```bash
python -m experiments.benchmark_tps --model custom --gamma 5 --max_new_tokens 512
```

`benchmark_tps.py` arguments:

| Argument | Type | Default | Constraints | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--model` | str | `custom` | one of {`custom`, `pythia`, `SmolLM`, `SmolLM2`} | Model family to be benchmarked |
| `--gamma` | int | 5 | > 0 | Baseline $\gamma$ for isolated tests before the sweep. |
| `--max_new_tokens` | int | 512 | > 0 | Maximum number of tokens to generate per prompt |

The script will:
- Instantiate the main model and draft models and load their saved checkpoints.
- Calculate **TPS (Tokens Per Second)** for the following configurations: 
   1. Main model (with/without cache)
   2. Draft model (with/without cache)
   3. Speculative engine (with/without cache)
- Calculate speedup (with/without cache) by comparing the TPS of the main model and speculative engine.
- Perform a $\gamma$ sweep ($\gamma \in \{1, 2, 3, 5, 7, 10\}$) and store the results in a dataframe.
- Perform a context-length stress test (e.g., 32 to 512 tokens) and store the results.

*Outputs are saved to `experiments/results/benchmarks_{model}.csv` and `experiments/results/stress_test_{model}.csv`.*

---

## 3. Plotting Systems Visualizations

Generate comparative latency distribution curves and speedup trajectories from the benchmarked CSV data.

```bash
python -m experiments.plot_graphs
```

The script will:
- Load dataframes containing the benchmark and profiler results from `experiments/results/`.
- Plot different graphs and store them in the `experiments/plots/` directory.

Generated plots include:
* `baseline_tps.pdf`: Baseline throughput metrics across model configurations.
* `smollm_profile.pdf`: Latency breakdown by execution stage vs. $\gamma$.
* `speedup_gamma.pdf`: Speedup curves demonstrating empirical peak $\gamma$ frontiers.
* `stress_test.pdf`: Throughput degradation as context expands.
* `ratio_vs_speedup.pdf`: Draft-to-main TPS ratio vs observed speedup.

---

## 4. Generation Methods

You can generate text using either standard autoregressive decoding or the speculative decoding engine directly.

### Standard Generation

```bash
python -m inference.generate --model main --max_new_tokens 512
```

`generate.py` arguments:

| Argument | Type | Default | Constraints | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--model` | str | `main` | one of {`main`, `draft_small`, `draft_medium`, `pythia-1B`, `pythia-160M`, `SmolLM-1.7B`, `SmolLM-135M`, `SmolLM2-1.7B`, `SmolLM2-135M`} | Model which should be used for generation |
| `--max_new_tokens` | int | 512 | > 0 | Maximum number of tokens to generate |
| `--no_cache` | flag | False | present or absent | Disable KV Cache during generation |

The script will:
- Instantiate the specified model and load the corresponding checkpoint.
- Tokenize the input prompt.
- Generate `max_new_tokens` number of output tokens.

### Speculative Decoding

```bash
python -m inference.speculative_engine --main_model main --draft_model draft_medium --gamma 5 --max_new_tokens 512 --return_stats
```

`speculative_engine.py` arguments:

| Argument | Type | Default | Constraints | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--main_model` | str | `main` | one of {`main`, `pythia-1B`, `SmolLM-1.7B`, `SmolLM2-1.7B`} | Main model to be used for verification of draft tokens |
| `--draft_model` | str | `draft_medium` | one of {`draft_small`, `draft_medium`, `pythia-160M`, `SmolLM-135M`, `SmolLM2-135M`} | Draft model to be used for speculative generation |
| `--gamma` | int | 5 | > 0 | Number of draft tokens to speculate per step |
| `--max_new_tokens` | int | 512 | > 0 | Maximum number of tokens to generate |
| `--no_cache` | flag | False | present or absent | Disable KV Cache across both engines |
| `--return_stats` | flag | False | present or absent | Return metrics such as empirical token acceptance rate and mean accepted tokens per round |

---

## 5. Evaluate Alignment

Evaluate the statistical alignment and token distribution overlap between a target model and a draft model.

```bash
python -m experiments.evaluate_alignment --draft_model draft_medium
```

`evaluate_alignment.py` arguments:

| Argument | Type | Default | Constraints | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--draft_model` | str | `draft_medium` | one of {`draft_small`, `draft_medium`} | Draft model to be used for evaluating alignment with main model |

The script will:
- Instantiate the main model and the draft model specified by `--draft_model`.
- Sample *n_batches* of validation data stream to evaluate the alignment score between the two models.

---

## 6. Training Custom Models (Optional)

If you wish to evaluate statistical alignment using the custom architectures trained from scratch, prepare the data and train using the following scripts.

### Prepare your data

```bash
python data/prepare_data.py
```

The script will:
- Load the *openwebtext* dataset using the Hugging Face `datasets` library.
- Use the pretrained *GPT-2* tokenizer to tokenize the text data.
- Create streaming memory-mapped binary train and validation data splits (`train.bin`, `val.bin`).

### Train the model

```bash
python train.py --model main
```

`train.py` arguments:

| Argument | Type | Default | Constraints | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--model` | str | `main` | one of {`main`, `draft_small`, `draft_medium`} | Model to train |

The script will:
- Instantiate the model specified by `--model`.
- Save the model configuration to `saved_models/{model_name}_config.json`.
- Resume training from an existing checkpoint (if available).
- Train for `max_iters` iterations.
- Evaluate on train/val splits every `eval_interval` steps.
- Save the best model as `saved_models/{model_name}_model.pt`.
- Generate sample text using a predefined prompt every 10,000 steps.

---

## Customization

Edit hyperparameters in `data/data_loader.py` at the top of the file:

```python
context_length = 1024
batch_size = 16
```

Edit hyperparameters in `data/prepare_data.py` at the top of the file:

```python
TRAIN_TOKENS = 500_000_000
VAL_TOKENS = 5_000_000
```

Edit hyperparameters in `train.py` at the top of the file:

```python
max_iters = 40_000
warmup_steps = 2_000
eval_iters = 20
eval_interval = 2_000
accumulation_steps = 16
base_lr = 3e-4
weight_decay = 0.1
```

---

## License

This project is licensed under the [MIT License](LICENSE).