# Book Recommendation System 
Part I of the project for E4571 Personalization: Theory & Application
Team members: Deepak Maran, Kewei Liu, Rakshita, Xiaohui Guo

One Paragraph of project description goes here.
Treat this as a case study and think about it from a business perspective. What is your objective? What are you trying to optimizing and what are you willing to sacrifice?

## Project Abstract
How well was this envisioned as an actual business problem? Is it clear what is being solved, and what acceptance criteria would be?

### The main contents of this project are following:
- Implementation of various algorithms
- Different ways of ensemble learning of the algorithms
- Support of parallel tasks on distributed system, and performance/time comparison of each algorithm by the number of computation instances
- Implementation of demo program of the final recommender system

### Algorithms to be implemented:
- Content-based Recommender System
- Collaborative Filtering
- Weight Learning
- Latent Factor Model
- Bias Extension
- Ensemble Model 

### Datasets
- Amazon Review Data accessed from (http://snap.stanford.edu/data/web-Amazon-links.html)
- Number of reviews : 34,686,770
- Number of users : 6,643,669 (56,772 with more than 50 reviews)
- Number of products : 2,441,053

### Tools
- Python, Myria, AWS


### Progress

1. Download data
You need to have 'data' directory in your HOME.

```bash
chmod +x download.sh; ./download.sh 
```

This will takes several minutes.

2. Preprocess data and create DB
```bash
python prepro/preprocess.py
```

3. Run Recommender System
```bash
python -m model.main
```
It builds recommender with train data and also evaluates performance on test data. If you want to specift certain recommender system, you can use '--recom'.

Content Based : 'cb'
Collaborative Filtering : 'cf'
Weight Learned : 'l'
Latent Factor : 'lf'
Latent Factor with Bias Extension : 'blf'

For example, if you want to run Weight Learned Recommender,
```bash
python -m model.main --recom l
```
If you want to run Content Based and Collaborative Filtering,
```bash
python -m model.main --recom cb cf
```
It runs recommender in small dataset by default. If you want to run in large dataset, you can use '--small False'. Batch size is 128 by default. If you want to change it, you can use '--batch_size'. For example,
```bash
python -m model.main --small False --batch_size 256
```

## References



