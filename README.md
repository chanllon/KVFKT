# KVFKT: A New Horizon in Knowledge Tracing with Attention-Based Embedding and Forgetting Curve Integration

<img src="KVFKTModelStructure.png" alt="drawing" width = "2000"> 

This is the official implementation for our paper **KVFKT: A New Horizon in Knowledge Tracing with Attention-Based Embedding and Forgetting Curve Integration**, accepted by **COLING'25**.




## Requirements
The code is built on Pytorch and the [pyKT](https://github.com/pykt-team/pykt-toolkit/tree/main) benchmark library. Run the following code to satisfy the requeiremnts by pip: `pip install -r requirements.txt`


## Datasets
- Download the three public datasets we use in the paper at:

  [ASSISTments 2012](https://sites.google.com/site/assistmentsdata/datasets/2012-13-school-data-with-affect)

  [EdNet](https://github.com/riiid/ednet)

  [NeurIPS](https://eedi.com/projects/neurips-education-challenge)
  
  [FSAI-F1toF3](https://www.4littletrees.com/)

- Preprocess the dataset using [pyKT](https://github.com/pykt-team/pykt-toolkit/tree/main) to obtain the student's mastery level of knowledge concepts **(MLKC)**, the probability of knowledge concepts appearing in the next exercise **(PKC)**, and the forgetting rate of knowledge concepts **(FRKC)**.


## Run KVFKT

1. Construct the knowledge graph: use pyKT preprocessed files, for example, `Algebra2005_head_10.csv`, to construct `entities.dict` (entity dictionaries), `relations.dict` (relationship dictionaries), `triples.txt` (triples required for knowledge graphs) and `Q.txt` (Q-matrix). Place the three generated files in folder `KG4Ex/data/algebra2005`.

2. Embedding learning: `python run.py --do_train --cuda --data_path ../data/algebra2005 --model TransE -b 1024 -d 1000 -g 12.0 -a 1.0 -lr 0.001 -adv -save models/algebra2005/TransE_adv`.

3. Recommendation and evaluation: the embedding vectors of entities and relations are saved in `KG4Ex/codes/models/algebra2005/TransE_adv`. Run `test_TransE.py` to obtain corresponding indicator results.


## The interpretability of KG4Ex
To validate the interpretability of KG4Ex and the rationality of exercise recommendations, we conducted real interviews with 250 real students. The student interviews were conducted through questionnaire surveys. We are making the questionnaire content public here `questionnaire.txt`.
