# isye6740-homework-4-solved
**TO GET THIS SOLUTION VISIT:** [ISYE6740-Homework 4 Solved](https://www.ankitcodinghub.com/product/isye6740-homework-4-solved/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;79404&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;3&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (3 votes)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;ISYE6740-Homework 4 Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (3 votes)    </div>
    </div>
<h1>1.&nbsp;&nbsp;&nbsp; Comparing classifiers.</h1>
In lectures, we learn different classifiers. This question is compare them on two datasets. Python users, please feel free to use Scikit-learn, which is a commonly-used and powerful Python library with various machine learning tools. But you can also use other similar libraries in other languages of your choice to perform the tasks.

<ol>
<li>Part One (Divorce classification/prediction).</li>
</ol>
This dataset is about participants who completed the personal information form and a divorce predictors scale. The data is a modified version of the publicly available at https://archive.ics.uci. edu/ml/datasets/Divorce+Predictors+data+set (by injecting noise so you will not get the exactly same results as on UCI website). The dataset <strong>marriage.csv </strong>is contained in the homework folder. There are 170 participants and 54 attributes (or predictor variables) that are all real-valued. The last column of the CSV file is label <em>y </em>(1 means “divorce”, 0 means “no divorce”). Each column is for one feature (predictor variable), and each row is a sample (participant). A detailed explanation for each feature (predictor variable) can be found at the website link above. Our goal is to build a classifier using training data, such that given a test sample, we can classify (or essentially predict) whether its label is 0 (“no divorce”) or 1 (“divorce”).

We are going to compare the following classifiers (<strong>Naive Bayes, Logistic Regression, and KNN</strong>). Use the first 80% data for training and the remaining 20% for testing. If you use scikit-learn you can use train test split to split the dataset.

<em>Remark: Please note that, here, for Naive Bayes, this means that we have to estimate the variance for each individual feature from training data. When estimating the variance, if the variance is zero to close to zero (meaning that there is very little variability in the feature), you can set the variance to be a small number, e.g.,</em><em>. We do not want to have include zero or nearly variance in Naive Bayes. This tip holds for both Part One and Part Two of this question.</em>

<ul>
<li>Report testing accuracy for each of the three classifiers. Comment on their performance: which performs the best and make a guess why they perform the best in this setting.</li>
<li>Now perform PCA to project the data into two-dimensional space. Build the classifiers(<strong>Naive Bayes, Logistic Regression, and KNN</strong>) using the two-dimensional PCA results. Plot the data points and decision boundary of each classifier in the two-dimensional space. Comment on the difference between the decision boundary for the three classifiers. Please clearly represent the data points with different labels using different colors.</li>
</ul>
<ol start="2">
<li>Part Two (Handwritten digits classification).</li>
</ol>
This question is to compare different classifiers and their performance for multi-class classifications on the complete MNIST dataset at http://yann.lecun.com/exdb/mnist/. You can find the data file <strong>mnist 10digits.mat </strong>in the homework folder. The MNIST database of handwritten digits has a training set of 60,000 examples, and a test set of 10,000 examples. We will compare <strong>KNN, logistic regression, SVM, kernel SVM, and neural networks</strong>.

<ul>
<li>We suggest you to “standardize” the features before training the classifiers, by dividing the values of the features by 255 (thus map the range of the features from [0, 255] to [0, 1]).</li>
<li>You may adjust the number of neighbors <em>K </em>used in KNN to have a reasonable result (you may use cross validation but it is not required; any reasonable tuning to get good result is acceptable).</li>
<li>You may use a neural networks function neural network with hidden layer sizes = (20, 10).</li>
<li>For kernel SVM, you may use radial basis function kernel and choose proper kernel.</li>
<li>For KNN and SVM, you can randomly downsample the training data to size <em>m </em>= 5000, to improve computation efficiency.</li>
</ul>
Train the classifiers on training dataset and evaluate on the test dataset.

<ul>
<li>Report confusion matrix, precision, recall, and F-1 score for each of the classifiers. Forprecision, recall, and F-1 score of each classifier, we will need to report these for each of the digits. So you can create a table for this. For this question, each of the 5 classifier, <strong>KNN, logistic regression, SVM, kernel SVM, and neural networks</strong>, accounts for 10 points.</li>
<li>Comment on the performance of the classifier and give your explanation why some ofthem perform better than the others.</li>
</ul>
<h1>2.&nbsp;&nbsp;&nbsp; Naive Bayes for spam filtering.</h1>
In this problem, we will use the Naive Bayes algorithm to fit a spam filter by hand. This will enhance your understanding to Bayes classifier and build intuition. This question does not involve any programming but only derivation and hand calculation.

Spam filters are used in all email services to classify received emails as “Spam” or “Not Spam”. A simple approach involves maintaining a vocabulary of words that commonly occur in “Spam” emails and classifying an email as “Spam” if the number of words from the dictionary that are present in the email is over a certain threshold. We are given the vocabulary consists of 15 words

<em>V </em>= {secret, offer, low, price, valued, customer, today, dollar, million, sports, is, for, play, healthy, pizza}<em>.</em>

We will use <em>V<sub>i </sub></em>to represent the <em>i</em>th word in <em>V </em>. As our training dataset, we are also given 3 example spam messages,

<ul>
<li>million dollar offer</li>
<li>secret offer today</li>
<li>secret is secret and 3 example non-spam messages</li>
<li>low price for valued customer</li>
<li>play secret sports today</li>
<li>low price pizza</li>
</ul>
Recall that the Naive Bayes classifier assumes the probability of an input depends on its input feature.

The feature for each sample is defined as &nbsp;and the class of the <em>i</em>th sample is <em>y</em><sup>(<em>i</em>)</sup>. In our case the length of the input vector is <em>d </em>= 15, which is equal to the number of words in the vocabulary <em>V </em>. Each entry is equal to the number of times word <em>V<sub>j </sub></em>occurs in the <em>i</em>-th message.

<ol>
<li>Calculate class prior P(<em>y </em>= 0) and P(<em>y </em>= 1) from the training data, where <em>y </em>= 0 corresponds to spam messages, and <em>y </em>= 1 corresponds to non-spam messages. Note that these class prior essentially corresponds to the frequency of each class in the training sample. Write down the feature vectors for each spam and non-spam messages.</li>
<li>In the Naive Bayes model, assuming the keywords are independent of each other (this is a simplification), the likelihood of a sentence with its feature vector <em>x </em>given a class <em>c </em>is given by</li>
</ol>
<em>d</em>

P(<em>x</em>|<em>y </em>= <em>c</em>) = <sup>Y </sup><em>θ<sub>c,k</sub><sup>x</sup></em><em><sup>k </sup>,&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; c </em>= {0<em>,</em>1}

<em>k</em>=1

where 0 ≤ <em>θ<sub>c,k </sub></em>≤ 1 is the probability of word <em>k </em>appearing in class <em>c</em>, which satisfies

<em>.</em>

Given this, the complete log-likelihood function for our training data is given by

Calculate the maximum likelihood estimates of <em>θ</em><sub>0<em>,</em>1</sub>, <em>θ</em><sub>0<em>,</em>7</sub>, <em>θ</em><sub>1<em>,</em>1</sub>, <em>θ</em><sub>1<em>,</em>15 </sub>by maximizing the log-likelihood function above.

(Hint: We are solving a constrained maximization problem and you will need to introduce Lagrangian multipliers and consider the Lagrangian function.)

<ol start="3">
<li>Given a test message “today is secret”, using the Naive Bayes classier that you have trained in Part (a)-(b), to calculate the posterior and decide whether it is spam or not spam.</li>
</ol>
<h1>3.&nbsp;&nbsp;&nbsp; Neural networks.</h1>
Consider a simple two-layer network in the lecture slides. Given <em>n </em>training data (<em>x<sup>i</sup>,y<sup>i</sup></em>), <em>i </em>= 1<em>,…,n</em>, the cost function used to training the neural networks

where <em>σ</em>(<em>x</em>) = 1<em>/</em>(1 + <em>e</em><sup>−<em>x</em></sup>) is the sigmoid function, <em>z<sup>i </sup></em>is a two-dimensional vector such that), and ). Show the that the gradient is given by

<em>,</em>

where <em>u<sup>i </sup></em>= <em>w<sup>T</sup>z<sup>i</sup></em>. Also show the gradient of <em>`</em>(<em>w,α,β</em>) with respect to <em>α </em>and <em>β </em>and write down their expression.
