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

`python main.py --dataset ***  --memory_size *** --value_memory_state_dim ***  --key_memory_state_dim  *** `.

For example

`python main.py --dataset NeurIPS  --memory_size 100 --value_memory_state_dim 200  --key_memory_state_dim  200 `.

## The interpretability of KVFKT
we demonstrate that KVFKT can capture realistic student knowledge states across multiple concepts during the learning process. To achieve this, we randomly select a student from the ASSIST2012 dataset and then evaluate the transition of the student's ability level, guess coefficient, and the probability of correctly answering the next KC as they progress in their learning by visualizing the first 30 attempts of that student.
<img src="Interpretability.pdf" alt="drawing" width = "2000"> 

# Citation

If you find our work helpful, please kindly cite our research paper:

@inproceedings{guan2025kvfkt,
  title={KVFKT: A New Horizon in Knowledge Tracing with Attention-Based Embedding and Forgetting Curve Integration},
  author={Guan, Quanlong and Duan, Xiuliang and Bian, Kaiquan and Chen, Guanliang and Huang, Jianbo and Gong, Zhiguo and Fang, Liangda},
  booktitle={Proceedings of the 31st International Conference on Computational Linguistics},
  pages={4399--4409},
  year={2025}
}


