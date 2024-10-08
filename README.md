# Naive-Bayes-Classifier
My own implementation on well-known probability method - Naive Bayes Classifier

This way of classification rely on the Bayes Theorem of dependent probability:
<figure style="float:display:block; margin-left: auto; margin-right: auto;">
	<img src="photos/BayesFormula.jpg" alt="" width="100%">
</figure>
<div style="clear:both">

By this, we can transform our probability into this:
<figure style="float:display:block; margin-left: auto; margin-right: auto;">
	<img src="photos/ClassBayes.jpg" alt="" width="100%">
</figure>
<div style="clear:both">

Moreover, if we suppose that all our features are independent we can expect that:
<figure style="float:display:block; margin-left: auto; margin-right: auto;">
	<img src="photos/big_formula.jpg" alt="" width="100%">
</figure>
<div style="clear:both">

Of course, this is quite confident assumption that all these features are independent. In real life we have a correlation between
features. That's why our classificator is called NAIVE. And it is assumed that this classificator is used as a baseline or as
a solution for small and simple datasets. 

If you want to use my classificator, you can not think about scaling of categorical features using OHE, or numerical features, using StandarScaler.
This is all because i do it by myself inside. But you can vary it by set transform=False. Also, all numeric features count their probality by
Gaussian formula:

<figure style="float:display:block; margin-left: auto; margin-right: auto;">
	<img src="photos/gaussian.jpg" alt="" width="100%">
</figure>
<div style="clear:both">

- x_i - the specific value of numeric featur
- sigma - variation of feature that has same target as X_i
- mean - mean of feature that has same target as X_i

## Dataset

I use adult dataset that has information about the adults and their income. 
The task is to predict - whether a person has income more than 50k USD, or less.
My classifier has 0.83 accuracy and 0.57 log_loss that is quite impressive for this simple and fast method.

## Current problems:
I want to adopt my implementation for a big number of features with sparse.
I tried to classify spam on text, but unfortunately it didn't work, because number of features is exremely huge.
This is also a big problem because all data become numeric and a little bit too way different.
That's why i recommend to use my implemetation if you have many categorical features and not too much numeric features.
It is also better that variation of each numeric feature be small.
[Kaggle-link to spam dataset](https://www.kaggle.com/datasets/venky73/spam-mails-dataset/data)!
