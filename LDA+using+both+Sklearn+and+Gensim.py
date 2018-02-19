
# coding: utf-8

# In[ ]:


import pandas as pd


# ### LDA using Sklearn

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation


# In[ ]:


email_domain = pd.read_csv("/Users/deepikamulchandani/Downloads/DataSets2_10012017/email_onlypersonaldomains.csv")


# In[ ]:


tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                   max_features=1000,
                                   stop_words='english')


# In[ ]:


tfidf = tfidf_vectorizer.fit_transform(email_domain['content'])


# In[ ]:


lda = LatentDirichletAllocation(n_components=30, max_iter=100,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)


# In[ ]:


lda.fit(tfidf)


# In[ ]:


tf_feature_names = tfidf_vectorizer.get_feature_names()


# In[ ]:


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()


# In[ ]:


print_top_words(lda,tf_feature_names,5)


# ### LDA using Gensim

# In[ ]:


email = pd.read_csv("/Users/deepikamulchandani/Downloads/DataSets2_10012017/email_persisting_employees.csv")


# In[ ]:


content = email['content']


# In[ ]:


import nltk
nltk.download('wordnet')
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string


# In[ ]:


stop = set(stopwords.words('english'))
exclude = set(string.punctuation) 
lemma = WordNetLemmatizer()
def clean(cont):
    stop_free = " ".join([i for i in cont.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

content_clean = [clean(cont).split() for cont in content]


# In[ ]:


import gensim
from gensim import corpora


# In[ ]:


dictionary = corpora.Dictionary(content_clean)


# In[ ]:


content_term_matrix = [dictionary.doc2bow(cont) for cont in content_clean]


# In[ ]:


Lda = gensim.models.ldamodel.LdaModel


# In[ ]:


ldamodel = Lda(content_term_matrix, num_topics=30, id2word = dictionary, passes=1, chunksize = 2000, iterations = 100)


# In[ ]:


print(ldamodel.print_topics(num_topics=30, num_words=5))

