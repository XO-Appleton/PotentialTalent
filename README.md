# Potential Talent

## Problem Statement

Rank candidates' qualification for an HR job based on their job title, location and connection number. Develop such an algorithm to generate initial rankings and rerank candidates every time a candidate is selected. The algorithm should be fully automized to avoid human bias as well as being able to be applied when selecting other roles.

## Data Description

Given the banking information of customers, predict if they would subscribe to a term deposit. Also identify the potential subscribers from the existing records.The data comes from our sourcing efforts. We removed any field that could directly reveal personal details and gave a unique identifier for each candidate.

Attributes:
id : unique identifier for candidate (numeric)

job_title : job title for candidate (text)

location : geographical location for candidate (text)

connections: number of connections candidate has, 500+ means over 500 (text)

Output (desired target):
fit - how fit the candidate is for the role? (numeric, probability between 0-1)

## Approach

Out of the three features that we have to use, location seem to be the least relevant in determining candidate's qualification. To verify our concern, we processed the location and plotted the candidate's distribution. Despite different distribution, it does not seem to correlate with the qualification of the candidates.

Candidate Distriution:

![candidate_distribution](https://github.com/XO-Appleton/PotentialTalent/assets/41369365/89b9c133-195a-4cb6-9877-99dc658ac8d7)

We then designed the ranking algorithm. We convert the job titles and target keywords into numerical vectors with pretrained BERT embeddings, then calculate their cosine similarities. Considering it is an HR job, having more connections should be considered an asset. Therefore, we engineered a new feature from connections called connection factors which has a value of 0 to 1 and multiplied it with the cosine similarities to get the initial ranking.

Now that we have an initial ranking to work with, we starred some of the top candidates. Then we fit an lightgbmranker model on the rankings which can then update the rankings.

## Metrics

We chose nDCG@10 for this project as it is a ranking problem. Notably the nDCG function from sklearn is not compatible with GridSearch and lightgbmranker, so we had to implement the metric on our own.

## Final Model

With GridSearch, we found the best model to be LGBMRanker(learning_rate=0.01, max_depth=5, metric='ndcg', min_child_samples=1, n_estimators=10, num_leaves=20, objective='lambdarank'). After training the model on the initial rankings, it is able to generate new rankings when new candidates are starred.

## Summary

In summary, we developped an algorithm to rank the job candidates based on their information without introducing human bias:

- We first converted the job titles of the candidates into numerical representations using pretrained BERT embedding and model.
- With the numerical representations, we are then able to compare the job titles of the candidates to the keywords to the job, and quantify such comparison by calculating the cosine similarity score.
- Combining the similarity score to the number of connections a candidate has, we are able to perform the initial ranking to the candidates.
- From the initial ranking, we randomly starred the top 40 candidates to be a good fit for the job.
- We then moved on to develop a ranking model leveraging lightgbm's ranker model using ndcg_score as metrics. To tune the model to the best configuration, we implemented a customized ndcg_score function and performed a GridSearch in the hyperparameter space.
- After building the model, we defined a function which allows the model to be retrained on new/updated data and generate rankings based on it.
- Based on the output of the retrained model. It seems that a good cut off threshold would be 0.15. Which means that we would filter out the candidates that have a negative final ranking. We could apply this algorithm to ranking the candidates for the other roles as well.
