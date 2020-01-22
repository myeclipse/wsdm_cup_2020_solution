# wsdm_cup_2020_solution
## Team name: ferryman
team member: Seiya,eclipse,will,ferryman


## 1. OVERVIEW
This repository contains our solution for [Citation Intent Recognition](https://biendata.com/competition/wsdm2020/), one of [WSDM Cup 2020](http://www.wsdm-conference.org/2020/wsdm-cup-2020.php) tasks.</br>

The competition provided a large paper dataset, which contains roughly 800K papers, along with paragraphs or sentences which describe the research papers. These pieces of description are mainly from paper text which introduces citations. The participants are required to recognize the paper cited in the describe texts.</br>

After analyzing the challenge, we regard it as an Information Retrieve (IR) task, The IR focuses on the problem of finding the most matched Top N documents with a query from a massive number of candidate documents. In this challenge, the description text is the query and the candidate papers are the documents to be retrieved. To handle this challenge, we made a plan with two stages including recall and ranking. In recall stage, several unsupervised methods are built to reduce the scope of candidates, then we draw learning to rank models to ranking the candidate papers which is selected in the recalling stage.</br>
## 2. RECALL STRAE
The recall results is not only used to reduce the retrieve scope for the rank model, but all as part of features used in the LGB ranking stage.

### 2.1 Text representation 
In the recall stage, candidate papers and descriptions were represented as a vector using vector space model and bag-of-N-gram model, in practice, the max N is set up to two owing to the huge computational space. <br/>

### 2.2 Retrieve strategy 
We use several similarity measurement to reduce the retrieve scope, including TFIDF, BM25, LM Dirichlet, AxiomaticF3EXP, DFI Similarity, AxiomaticF1EXP, AxiomaticF2EXP, AxiomaticF1LOG, AxiomaticF2LOG, AxiomaticF3LOG, Boolean Similarity, LM Jelinek Mercer Similarity, DFR Similarity, IB Similarity and so on. And we apply the structure introduced above on different scales of a paper, such as title, abstract, keywords and full text.<br/>
To accelerate the retrieve speed, inverted index was employed in our solution. An inverted index is an index data structure storing a mapping from content, such as words or numbers, to its locations in a document or a set of documents.

### 2.3 Compare & analysis
In our practice, the F1EXP has the highest recall score and BM25 get the highest MAP score.<br/>

## 3 RANK STRATAGE
### 3.1 Bert model
#### 3.1.1 Bert model
The BERT model architecture is based on a multilayer bidirectional Transformer. Instead of the traditional left-to-right language modeling objective, BERT is trained on two tasks: predicting randomly masked tokens and predicting whether two sentences follow each other. BERT model gets a lot of state of the arts in many tasks, and we also use the BERT model. There are two type of BERT models follows the same architecture as BERT but instead pretrained on different scientific text: SciBERT and BioBERT. Also, we trained the pretrained model in two ways: Point-Wise model and Pair-Wise model.<br/>
#### 3.1.2 Data preprocessing 
The input of the BERT model determines the upper bound of the score. We have to preprocess the sentence to try to take the highest advantage of the model. Firstly, we remove the excess whitespace and some stop words in English, do some word segmentation and do part-of-speech tagging. Secondly, we normalized the word form for different tag of the sentence. And finally, we lowercased all letters. We compared the input without preprocessing and the input with preprocessing, and found that the input with preprocessing is better than the other one.<br/>

#### 3.1.3 Bert Model With Point-wise
We trained the BERT model in Point-Wise way which means we defined the task as the binary classification. We preprocessed the two sentences (the description sentence and the paper-described sentence) and with [SEP] token, we joined them in one sentence and put them into the BERT model. We trained the token of sentence with binary cross entropy loss to dig the difference between description sentence and paper-described sentence. We want to use the probability to measure how well the two sentences match. However, lots of negative samples can destroy the performance of the BERT model and the pointwise class method does not take into account the internal dependencies between the docs corresponding to the same query. On the one hand, the samples in the input space are not IID, which violates the basic assumptions of ML. On the other hand, the structure between these samples is not fully utilized. Second, when different queries correspond to different numbers of docs, the overall loss will be dominated by the query group with a large number of docs. Each group of queries should be equivalent. Do we have a better way to get a better performance of the model? We tried the Pair-Wise model.<br/>

#### 3.1.4 Bert model with pair-wise
Learning2Rank applies ML technology to the ranking problem and trains the ranking model. Usually the discriminant supervised ML algorithm is applied. The metrics mentioned in the overview should be based on query and position, Learning2Rank task seeks ranking results, and does not require precise scoring, as long as there is a relative scoring. Pairwise class methods, whose L2R framework has the following characteristics:<br/>
1.	The samples in the input space are two feature vectors (corresponding to the same query) composed of two doc (and corresponding query).
2.	The samples in the output space are pairwise preference.
3.	The samples in the space are two-variable functions and the loss function evaluates the difference between the predicted preference and the true preference of the doc pair.
We do the same preprocessing to the input sentence as the way described in the above. However, we put the one true pair and one negative pair in one sample. We use the margin ranking loss as our loss function. We trained several triplet samples with the same description text and different paper-described sentences. It not only helps to get a better ranking of similarity, but also compares the differences between each description text. We got a higher score than the BERT model with Point-Wise.<br/>


### 3.2 Lightgbm model
In order to increase the diversity of the model, in addition to Bert, we choose Lightgbm for modeling, and for simplicity, it is called lgb here. And  compared with Bert, the effect of lgb is better. Total number of features is 1684, and the training method of Lgb is lambdarank, which is about 0.5% higher than the traditional binary classification model. The following will be carried out from two aspects of feature engineering and model construction:
#### 3.2.1 Feature  engineering
Our feature engineering mainly consists of the following 3 aspects:
1. Semantic features  
Semantic features include various pre-trained word vector models such as fasttext, glove, etc. And we retrain them to calculate the similarity between description and abstract.
2. Statistical features and word frequency features  
In this section, we use various word frequency-based methods to capture similarities, such as bm25, tfidf, f1exp and various length and proportion features.
3. Rank features  
In order to make our model easier to “know” the essential purpose of ranking, we sort the various similarity values according to description_id, and divide the ranking value by the number of description_id to get the relative ranking ratio. This part can bring a 3% boosting.

#### 3.2.2 Modeling methodology 
Our Lgb model is trained using a 5-fold cross-validation method. The training target is lambdarank, and the offline verification indicators are Map @ 3 and Map @ 5. 

## 4.ENSEMBLE METHODOLOGY
In the model ensemble stage, we adopted a simple and efficient way which based on blending. We group the model prediction results of lgb and bert by description_id, and then add the the ranking values with weighting opertion.
## 5. CONCLUSION
After our recall strategy and multi-model strategy, our final score in this competition is 0.425. We are grateful to the competition committee for organizing this competition, and thank the members for their efforts.<br/>
## REFERENCE
\[1\]	G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, and T.-Y. Liu. Lightgbm: A highly efficient gradient boosting decision tree. In Advances in Neural Information Processing Systems, pages 3149–3157, 2017.<br/>
\[2\]	Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. and Gulin, A. (2019). CatBoost: unbiased boosting with categorical features.Papers.nips.cc. Available at: http://papers.nips.cc/paper/7898-catboost-unbiased-boosting-with-categorical-features<br/>
\[3\]	Tianqi Chen and Carlos Guestrin. 2016. Xgboost: A scalable tree boosting system. In Proceedings of the 22nd acm sigkdd international conference on knowledge discovery and data mining. ACM, 785–794.<br/>


## APPENDIX
1. During the challenge, we found that only using the header candidate papers can accelerate the speed and achieve better results. The score on the final leaderboard only used the top 50000 candidate papers.
2. We also found an interest thing that the field 'journal' of all the test data are not 'no_content'.