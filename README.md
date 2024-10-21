# truthfulqa-multi
## Multilingual TruthfulQA

Obtain the answers of the model using harness:

```
sbatch experiments/generative.slurm
```

Judge the answers with the judge-model:

```
sbatch judge/run_experiments/judge.slurm
```

Evaluate the judges:

```
python judge/correlate_to_manual.py
```
