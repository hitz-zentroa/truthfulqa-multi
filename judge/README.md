
The instances for the manual revision datasets were obtained with ```judge/manual_review/get_manual_revision_instances.py```
The IAA of the evaluations was calculated with ```judge/manual_review/iaa_evaluation.py```
And stats with ```judge/manual_review/stats.py```

To translate the training data for the judge, we performed:
```
# put answers in txt format for the translation library
python judge/translate_training_data/to_txt_format.py
# translate using the easy-tranlsate library
sbatch judge/translate_training_data/traductor.slurm
# matched questions between languages and reconstructed dataset
python judge/translate_training_data/build_multilingual_train_set.py #TODO
```

To train the model:
```
sbatch judge/train_judge/finetune.slurm
```

To evaluate with the judge:
```
sbatch judge/train_judge/generate.slurm
```

To correlate the judge results to the manual evaluation:
```
python judge/correlate_to_manual.py # TODO
```