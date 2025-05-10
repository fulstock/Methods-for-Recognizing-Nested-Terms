# [Methods for Recognizing Nested Terms](https://arxiv.org/abs/2504.16007)

## Introduction

This is the repository for our ["Methods for Recognizing Nested Terms" paper](https://arxiv.org/abs/2504.16007), published in Computational Linguistics and Intellectual Technologies: Proceedings of the International Conference “Dialogue 2025”.

In this paper, we describe our participation in the RuTermEval competition devoted to extracting nested terms.
We apply the Binder model, which was previously successfully applied to the recognition of nested named entities,
to extract nested terms. We obtained the best results of term recognition in all three tracks of the RuTermEval
competition. In addition, we study the new task of recognition of nested terms from flat training data annotated
with terms without nestedness. We can conclude that several approaches we proposed in this work are viable
enough to retrieve nested terms effectively without nested labeling of them.

If you find our code is useful, please cite:
```
@misc{rozhkov2025methodsrecognizingnestedterms,
      title={Methods for Recognizing Nested Terms}, 
      author={Igor Rozhkov and Natalia Loukachevitch},
      year={2025},
      eprint={2504.16007},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2504.16007}, 
}
```

## Binder model

Binder model (with our fixes and additional featured scripts) used in our study can be found [here](https://github.com/fulstock/binder).

## RuTermEval dataset

RuTermEval dataset was created as part of RuTermEval competition, with three different tracks. You can find all three competitions here: [Track 1](https://codalab.lisn.upsaclay.fr/competitions/19597), [Track 2](https://codalab.lisn.upsaclay.fr/competitions/19600), [Track 3](https://codalab.lisn.upsaclay.fr/competitions/19601). 

RuTermEval organizers paper with details for mentioned dataset can be found [here](https://dialogue-conf.org/wp-content/uploads/2025/04/MamontovaAIschenkoRVorontsovK.pdf).

Train and development subsets can be downloaded through links above (see Participate -> Get Data or Files). Test subset is downloadable too, but gold labeling is not publicly available. You can reproduce our study or test your models on test subset through Codalab platform on all three tracks. Only registration is needed. 

## Scripts

All scripts were run on Python 3.10.11. You can find needed packages at `requirements.txt` and install them via `pip install -r requirements.txt`.

### Nested Term Recognition from Flat Supervision Approaches

#### Baselines

- `prepare_dataset.py`: convert RuTermEval format from Codalab to Binder format. This data corresponds to *full nested learning* approach described in our paper.
- `prepare_flat_dataset.py`: same as `prepare_dataset.py`, but only outermost flat data is preserved. Corresponds to *pure flat learning* approach.  
#### Non-damage approaches

- `prepare_winc_dataset.py`: *simple inclusions* approach.
- `prepare_lemwinc_dataset.py`: *lemmatized inclusions* approach.
- `prepare_lemwincdamage_dataset.py`: *lemmatized inclusions* fused with *damaged cross-prediction approach* (both *early* and *late*). Damaged cross-predicted data should be prepared before running this script, see details below.
#### Damaged cross-prediction approaches 

*Damaged cross-prediction* approaches can be found in `damage/early` and `damage/late` respectively. 

Both directories have same logical structure. They should be run in the following order:
1. `damage_data.py`: script to damage data (train for early, development and test for late);
2. `data4cross_predict.py`: script to prepare damaged data for cross-prediction. Obtained data can be used for subtraining;
3. `convert_and_merge.py`: fuse resulting predictions with flat data for final training.

### Other

Two other scripts are provided:
- `convert_for_submission.py`: convert Binder predictions to Codalab format for evaluation;
- `check_f1_nested.py`: calculate F1 inner, outer and overall metrics over macro- and micro-average scores.

## Processed data

Processed RuTermEval data for all approaches in Binder format can be found in `processed_data` directory.

## Contacts

For any inquires you can reach out to fulstocky@gmail.com.