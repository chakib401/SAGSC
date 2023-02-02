
# Scalable Attributed-Graph Subspace Clustering (SASGC)

This repository provides Python code to reproduce experiments from the AAAI 2023 paper *Scalable Attributed-Graph Subspace Clustering*.


## Run Experiments
#### Parameter List for `run.py`


| Parameter        | Type           | Default | Description  |
| :-------------: |:-------------:| :----:|:-------------------------------- |
| `dataset` | string| `acm`| Name of the graph dataset (`acm`, `dblp`, `arxiv`, `pubmed` or `wiki`). |
| `power` | integer| `2`| First power to test. |
| `runs` | integer| `5`| Number of runs. |

#### Best Propagation Orders


| Dataset        | Propagation order           |
| :-------------: |:-------------:|
|`acm`| `2`|
|`dblp`| `2`|
|`arxiv`| `54`|
|`computers`| `67`|
|`wiki`| `4`|
|`pubmed`| `100`|

#### Example

To run the model on computers for power `p=67` and have the average execution time
```bash
python run.py --dataset=computers --power 67
```

## Citation

If you use this code please do cite :

```BibTeX
@inproceedings{fettal2023scalable,
  author = {Fettal, Chakib and Labiod, Lazhar and Nadif, Mohamed},
  title = {Scalable Attributed-Graph Subspace Clustering},
  year = {2023},
  booktitle = {Proceedings of the 37th AAAI Conference on Artificial Intelligence}
}
```

