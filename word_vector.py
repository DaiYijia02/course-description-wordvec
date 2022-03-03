# for data
import pandas as pd
import numpy as np
# for plotting
import matplotlib.pyplot as plt
import seaborn as sns
# for processing
import re
import nltk
# for bag-of-words
from sklearn import feature_extraction, model_selection, naive_bayes, pipeline, manifold, preprocessing
# for explainer
from lime import lime_text

# read sp22
df = pd.read_csv("sp22.tsv", sep='\t')

'''
Preprocess a string.
:parameter
    :param text: string - name of column containing text
    :param lst_stopwords: list - list of stopwords to remove
    :param flg_stemm: bool - whether stemming is to be applied
    :param flg_lemm: bool - whether lemmitisation is to be applied
:return
    cleaned text
'''
def utils_preprocess_text(text):
    # clean (convert to lowercase and remove punctuations and characters and then strip)
    text = re.sub(r'[^\w\s-]', '', str(text).lower().strip())

    # Tokenize (convert from string to list)
    lst_text = text.split()

    # back to string from list
    text = " ".join(lst_text)
    return text


df["text_clean"] = df["Description"].apply(lambda x:
                                           utils_preprocess_text(x))

# list of eng and hum subjects, also common words
eng_dept = open('eng_dept.txt', 'r')
Eng = eng_dept.readlines()
Eng = [t.strip() for t in Eng]
hum_dept = open('hum_dept.txt', 'r')
Hum = hum_dept.readlines()
Hum = [t.strip() for t in Hum]
common_words = open('common_words.txt', 'r')
Dic = common_words.readlines()
Dic = [t.strip() for t in Dic]


'''
Change subjects to labels: eng -> 0, hum -> 1
'''
def label(subject):
    if (subject in Eng):
        return 0
    else:
        return 1


df["y"] = df["Subject"].apply(lambda x:
                              label(x))

# split dataset
df_train, df_test = model_selection.train_test_split(df, test_size=0.3)
# get target
y_train = df_train["y"].values
y_test = df_test["y"].values

# vectorize to word vector
vectorizer = feature_extraction.text.CountVectorizer(
    max_features=10000, vocabulary=Dic)

corpus = df_train["text_clean"]
vectorizer.fit(corpus)
X_train = vectorizer.transform(corpus)

# sns.heatmap(X_train.todense()[:, np.random.randint(0, X_train.shape[1], 100)]
#             == 0, vmin=0, vmax=1, cbar=False).set_title('Sparse Matrix Sample')

# top 10 words in engr and humi
X_engr = X_train[np.where(y_train == 0)]
X_humi = X_train[np.where(y_train == 1)]
sum_words_engr = X_engr.sum(axis=0)
sum_words_humi = X_humi.sum(axis=0) 
words_freq_engr = [(word, sum_words_engr[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
words_freq_humi = [(word, sum_words_humi[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
words_freq_engr =sorted(words_freq_engr, key = lambda x: x[1], reverse=True)
words_freq_humi =sorted(words_freq_humi, key = lambda x: x[1], reverse=True)
df1 = pd.DataFrame(words_freq_engr[:30], columns = ['Word', 'Engineering Count'])
df1.plot.bar(x='Word',y='Engineering Count',color='#153a9d')
df2 = pd.DataFrame(words_freq_humi[:30], columns = ['Word', 'Humanity Count'])
df2.plot.bar(x='Word',y='Humanity Count',color='#9d154c')


'''
Multinomial Naive Bayes
'''
classifier1 = naive_bayes.MultinomialNB()

'''
Linear Support Vector Machine
'''
from sklearn.svm import LinearSVC
classifier2 = LinearSVC(tol=1.0e-6,max_iter=10000,verbose=1)

'''
Logistic Regression
'''
from sklearn.linear_model import LogisticRegression
classifier3 = LogisticRegression(random_state=0).fit(X_train, y_train)

## pipeline
model1 = pipeline.Pipeline([("vectorizer", vectorizer),  
                           ("classifier", classifier1)])
model2 = pipeline.Pipeline([("vectorizer", vectorizer),  
                           ("classifier", classifier2)])
model3 = pipeline.Pipeline([("vectorizer", vectorizer),  
                           ("classifier", classifier3)])
## train classifier
model1["classifier"].fit(X_train, y_train)
model2["classifier"].fit(X_train, y_train)
model3["classifier"].fit(X_train, y_train)

## test
X_test = df_test["text_clean"].values
score1 = model1.score(X_test, y_test)
score2 = model2.score(X_test, y_test)
score3 = model3.score(X_test, y_test)

# find the y vector for MultinoialNB
fcount = model1.named_steps["classifier"].feature_log_prob_
y_engr1 = fcount[0]
y_humi1 = fcount[1]
y_engr_words1 = [(word, y_engr1[idx]) for word, idx in vectorizer.vocabulary_.items()]
y_engr_words1 =sorted(y_engr_words1, key = lambda x: x[1], reverse=True)
y_humi_words1 = [(word, y_humi1[idx]) for word, idx in vectorizer.vocabulary_.items()]
y_humi_words1 =sorted(y_humi_words1, key = lambda x: x[1], reverse=True)
df3 = pd.DataFrame(y_engr_words1[:30], columns = ['Word', 'Engineering Log Probability'])
df3.plot.bar(x='Word',y='Engineering Log Probability', color='#3e86e5')
df4 = pd.DataFrame(y_humi_words1[:30], columns = ['Word', 'Humanity Log Probability'])
df4.plot.bar(x='Word',y='Humanity Log Probability',color='#ed78bc')

# find y vector in LinearSVM
coef = model2.named_steps["classifier"].coef_
y_engr_words2 = [(word, coef[0,idx]) for word, idx in vectorizer.vocabulary_.items()]
y_engr_words2 =sorted(y_engr_words2, key = lambda x: x[1], reverse=True)
df5 = pd.DataFrame(y_engr_words2[:30], columns = ['Word', 'Engineering SVM'])
df5.plot.bar(x='Word',y='Engineering SVM', color='#78c8ed')

# find y vector for Logistic Regression
coef2 = model3.named_steps["classifier"].coef_
y_engr_words3 = [(word, coef2[0,idx]) for word, idx in vectorizer.vocabulary_.items()]
y_engr_words3 =sorted(y_engr_words3, key = lambda x: x[1], reverse=True)
df6 = pd.DataFrame(y_engr_words3[:30], columns = ['Word', 'Logistic Regression'])
df6.plot.bar(x='Word',y='Logistic Regression', color='#3ae4c5')

#find the class that is most and least confident
pred_prob=model3.predict_proba(df["text_clean"])
y_engr_words4 = [(df["Subject"][i]+str(df["Course Number"][i].item()), pred_prob[i,0]) for i in range(df["Subject"].size)]
y_engr_words4 =sorted(y_engr_words4, key = lambda x: x[1], reverse=True)
df7 = pd.DataFrame(y_engr_words4[:30], columns = ['Word', 'Engineering Most Confident'])
df7.plot.bar(x='Word',y='Engineering Most Confident', color='#1abc45')
df8 = pd.DataFrame(y_engr_words4[-30:], columns = ['Word', 'Engineering Least Confident'])
df8.plot.bar(x='Word',y='Engineering Least Confident', color='#bc1a1a')

y_humi_words4 = [(df["Subject"][i]+str(df["Course Number"][i].item()), pred_prob[i,1]) for i in range(df["Subject"].size)]
y_humi_words4 =sorted(y_humi_words4, key = lambda x: x[1], reverse=True)
df9 = pd.DataFrame(y_humi_words4[:30], columns = ['Word', 'Humanity Most Confident'])
df9.plot.bar(x='Word',y='Humanity Most Confident', color='#9fbc1a')
df10 = pd.DataFrame(y_humi_words4[-30:], columns = ['Word', 'Humanity Least Confident'])
df10.plot.bar(x='Word',y='Humanity Least Confident', color='#9566ea')


#confidence curve
pred_prob=model3.predict_proba(df["text_clean"])
df0 = pd.DataFrame({'Engineering':sorted(pred_prob[:,0]),'Humanity':sorted(pred_prob[:,1])})
df0['Engineering'] = df0['Engineering'].astype(float)
df0['Humanity'] = df0['Humanity'].astype(float)
df0.plot.line()
plt.show()
