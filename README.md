# Multilingual TruthfulQA

This repository contains the code and resources for the paper "Truth Knows No Language: Evaluating Truthfulness Beyond English".

## Abstract
We introduce a professionally translated extension of the TruthfulQA benchmark designed to evaluate truthfulness in Basque, Catalan, Galician, and Spanish. Truthfulness evaluations of large language models (LLMs) have primarily been conducted in English. However, the ability of LLMs to maintain truthfulness across languages remains under-explored. Our study evaluates 12 state-of-the-art open LLMs, comparing base and instruction-tuned models using human evaluation, multiple-choice metrics, and LLM-as-a-Judge scoring. Our findings reveal that, while LLMs perform best in English and worst in Basque (the lowest-resourced language), overall truthfulness discrepancies across languages are smaller than anticipated. Furthermore, we show that LLM-as-a-Judge correlates more closely with human judgments than multiple-choice metrics, and that informativeness plays a critical role in truthfulness assessment. Our results also indicate that machine translation provides a viable approach for extending truthfulness benchmarks to additional languages, offering a scalable alternative to professional translation. Finally, we observe that universal knowledge questions are better handled across languages than context- and time-dependent ones, highlighting the need for truthfulness evaluations that account for cultural and temporal variability.

## Resources

* **Paper:** [https://arxiv.org/abs/2502.09387](https://arxiv.org/abs/2502.09387)
* **Dataset:** [https://huggingface.co/datasets/HiTZ/truthfulqa-multi](https://huggingface.co/datasets/HiTZ/truthfulqa-multi)
* **Judges:** [Link to Hugging Face judges - to be added]

## Repository Structure

-   **`analysis/`**: Scripts and notebooks for analyzing the results of the experiments and evaluations.
-   **`data/`**: Contains the raw and processed data used for the experiments.
-   **`experiments/`**: Scripts to run the main experiments, including generating model answers.
-   **`judge/`**: Code related to the judge models, including running the judges and evaluating their performance.
-   **`MT_experiments/`**: Scripts and resources for experiments involving machine translation.
-   **`results/`**: Stores the outputs and saved results from various experiments and evaluations.
-   **`utils/`**: Utility scripts and helper functions used across the project.

## Usage

### Obtain model answers

Obtain the answers of the model using harness. These scripts typically call shell scripts located in `experiments/run/`.

For standard models:
```
sbatch experiments/generative.slurm
```

For larger models:
```
sbatch experiments/generative_big.slurm
```

### Run MC2

To run the MC2:

For standard models:
```bash
sbatch experiments/mc2.slurm
```

For larger models:
```bash
sbatch experiments/mc2_big.slurm
```

### Analyze Results

To generate statistics on judge performance by category and type:
```bash
python analysis/get_stats.py
```
This will output `by_category.csv` and `by_type.csv` in the `analysis/` directory.

To calculate inter-annotator agreement (Cohen's Kappa) between manual evaluations and MC2 results:
```bash
python analysis/iaa_mc2.py
```

### Compare to Machine Translated (MT) Data Analysis

Scripts in `analysis/compare_to_MT_translation/` are used to compare results obtained using human-translated data versus machine-translated data.

-   **`check_translation_quality.py`**: Evaluates the quality of machine-translated questions and answers against human translations using metrics like BLEU, CHRF, etc. It outputs scores to the console and saves any errors to `analysis/compare_to_MT_translation/errors.csv`.
    ```bash
    python analysis/compare_to_MT_translation/check_translation_quality.py
    ```
-   **`comparison.py`**: Compares judge outputs for human-translated data versus machine-translated data (from `judge/judge_output/MT-claude/`). It calculates agreement and identifies instances where judgments differ, saving these to `analysis/compare_to_MT_translation/check_instances.csv` and `analysis/compare_to_MT_translation/check_instances_3wrong.csv`.
    ```bash
    python analysis/compare_to_MT_translation/comparison.py
    ```
-   **`statistical_test_MT.py`**: Performs statistical tests (Chi-square) to compare the distributions of 'yes'/'no' judgments between human-translated and machine-translated datasets for different models and languages.
    ```bash
    python analysis/compare_to_MT_translation/statistical_test_MT.py
    ```
-   The `analysis/compare_to_MT_translation/translate/` directory contains scripts for performing machine translation, likely using different services/models (e.g., `translate_with_anthropic.py`).

### Cultural Nuances Analysis

Scripts in `analysis/cultural_nuances/` are used to investigate if there are differences in model performance on questions with local/cultural nuances versus global questions. It relies on a predefined list of culturally specific instances (likely from `veritasqa.csv`).

-   **`test_local_instances.py`**: Loads judge results, separates instances based on whether they are in the `veritasqa.csv` list (global) or not (local), and then calculates and outputs performance metrics for these two subsets. Results are saved to `analysis/cultural_nuances/results.csv`.
    ```bash
    python analysis/cultural_nuances/test_local_instances.py
    ```

### Judging Process

This project includes a comprehensive framework for judging the truthfulness of model-generated answers.

**1. Train Judge Models (Optional):**
   - If you need to train custom judge models, refer to the scripts and resources in `judge/train_judge/`.
   - Training data may involve translation, see `judge/translate_training_data/`.

**2. Judge Model Answers:**
   - Use the following command to have the judge model evaluate the answers generated in the previous steps. This typically calls scripts in `judge/run_experiments/`.
```bash
sbatch judge/run_experiments/judge.slurm
```

**3. Process Judge Results:**
   - The script `judge/judge_results.py` can be used to process or aggregate the outputs from the judge models.

**4. Evaluate Judges:**
   - Compare the judge model's evaluations against manual annotations:
```bash
python judge/correlate_to_manual.py
```
   - Further details and results related to judging can be found in `judge/README.md`, `judge/iaa_informativeness.txt`, and `judge/report_results.txt`.

### Run Machine Translation Experiments

To run the machine translation experiments, you can use the provided Slurm scripts:

For smaller models:
```bash
sbatch MT_experiments/generative.slurm
```

For larger models:
```bash
sbatch MT_experiments/generative_big.slurm
```
These scripts will execute the relevant experiment scripts located in the `MT_experiments/run/` directory.

Judge the answers with the judge-model:

```
sbatch judge/run_experiments/judge.slurm
```

Evaluate the judges:

```
python judge/correlate_to_manual.py
```
