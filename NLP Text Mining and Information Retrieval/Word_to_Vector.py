#importing required libraries
import pandas as pd
import numpy as np
import string
import nltk  #for word tokenization in pre-processing steps"
#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from scipy import spatial
from nltk.corpus import stopwords
stop = stopwords.words('english')

#from nltk.stem.snowball import SnowballStemmer
#stemmer = SnowballStemmer("english")

lemmatizer = nltk.stem.WordNetLemmatizer()
def word2vec_avg(Query_string):

    # Read the corpus
    data_corpus = pd.read_csv("articles1.csv", nrows=2000)
    data_corpus = data_corpus.drop(['publication','author','date','year','month','url','Unnamed: 0'],axis=1)
    data_corpus['con_title_content'] = data_corpus['title']+"."+data_corpus['content']
#    for col in data_corpus.columns:
#        print(col)      
#    data_corpus.head()
    #Appending the query string(input string for comparison, ideally Title or Random text) to the corpus.
    data_corpus=data_corpus.append(pd.Series(["999999999999","","",Query_string], index=data_corpus.columns),ignore_index=True)
#    data_corpus.tail() 
    data_frame = data_corpus.copy()    
       
    #Function to perform the pre-processing steps required on the Corpus.
    def preprocessing(data_frame, column):
        #Stop words removal
        data_frame[column] = data_frame[column].apply(lambda x:' '.join([i for i in word_tokenize(str(x)) 
                                                  if i not in stop]))
        #unnecessary punctuations removal
        data_frame[column] = data_frame[column].apply(lambda x:''.join([i for i in x 
                                                  if i not in string.punctuation]))
        #transform data into lower case
        data_frame[column] = data_frame[column].str.lower()
        #replace multiple spaces with single space
        data_frame[column] = data_frame[column].replace('( +)', r' ', regex=True)
        #stemming
#        data_frame[column] = data_frame[column].apply(lambda x:' '.join([stemmer.stem(y) 
#                                                                                       for y in word_tokenize(str(x))]))
        #lemmatization
        data_frame[column] = data_frame[column].apply(lambda x:' '.join([lemmatizer.lemmatize(w) 
                                                                                        for w in word_tokenize(str(x))]))
        return data_frame[column]
    
    data_frame['con_title_content'] = preprocessing(data_frame, 'con_title_content')
    
    #The below statement is for exporting the Pre-Processed Corpus to Excel Sheeet.
    #export_excel = data_frame.to_excel (r'C:\Users\ksingara\Documents\POC_NLP_Similarity\Article_Matching\export_dataframe.xlsx', index = None, header=True) #Note to add '.xlsx' at the end of the path
    
    #Storing the documents of the corpus in a list.
    list_desc = data_frame["con_title_content"].tolist()
    
    
    #preparing data for word2vec training
    tokens = []
    for sentence in list_desc:
        tokens.append(word_tokenize(sentence))
    
    #Training word2vector model    
    model = Word2Vec(tokens, size=200, window=9, min_count=1, workers=4)
    model.save("W2V_Model.model")
    
    model = Word2Vec.load("W2V_Model.model")

    #print the vocabulary on which the model got trained
    #w2c = dict()
    #for item in model.wv.vocab:
        #w2c[item]=model.wv.vocab[item].count
    #To find size of the vocab
    #print("########### Size of the Vocab################")
    #print(len(model.wv.vocab))
    #print("###########################")
    
    #function to form average word embeddings for title+contents
    index2word_set = set(model.wv.index2word)
    def avg_feature_vector(sentence, model, num_features, index2word_set):
        words = sentence.split()
        feature_vec = np.zeros((num_features, ), dtype='float32')
        n_words = 0
        for word in words:
            if word in index2word_set:
                n_words += 1
                feature_vec = np.add(feature_vec, model[word])
        if (n_words > 0):
            feature_vec = np.divide(feature_vec, n_words)
        return feature_vec

    #create average word embeddings for all title+contents
    desc_word2vec = []
    for document in data_frame["con_title_content"]:
        desc_word2vec.append(avg_feature_vector(document, model=model, num_features=200, index2word_set=index2word_set))
  
    #Function for finding the cosine similarities and getting top 10 documents with respect to their similarity scores.
    # In W2V, sice the matrix size is big, hence we shall be using the spatial distance cosine similarity
    def similar_desc_indexes(desc_word2vec,query_index):
        retrieval = []
        for i in range(len(desc_word2vec)-2):
            retrieval.append(1 - spatial.distance.cosine(desc_word2vec[query_index], desc_word2vec[i]))
        #retrive indices of top 5 similar articles
        related_product_indices = sorted(range(len(retrieval)), key=lambda i: retrieval[i])[:-11:-1]    
        #print(related_product_indices)
        result = pd.DataFrame(columns=['id','Score', 'title', 'content', 'con_title_content'])  
        for i in related_product_indices:        
            result = result.append(pd.Series([data_corpus["id"][i],round(retrieval[i]*100,3),data_corpus["title"][i],data_corpus["content"][i],data_corpus["con_title_content"][i]],index=result.columns),ignore_index=True)
        return result
    
    #Function for extracting the top 5 similarities with respect to the scores.
    def get_top_descriptions(result):   
        result.sort_values("Score", axis=0, ascending=False, inplace=True, kind='quicksort')         
        return result
    #Top 10 results with respect to Query String (title or random text)
    QuryStrresult = similar_desc_indexes(desc_word2vec,query_index = -1)
    result = QuryStrresult
    #Calling the function for getting top5 results out of Query String
    article_list = get_top_descriptions(result)

    #Converting the List to JSON sequential Objects.
    json=article_list[0:5].to_json(orient='index')
    
    #for printing vectors
#    def view_vectors(pre_processed_str,w2v_model):
#        dic = dict()
#        for word in word_tokenize(pre_processed_str):
#            if word in w2v_model.wv.vocab:
#                dic.update({word:w2v_model[word]})
#                print(word,":",w2v_model[word])
#                print()             
#                print(word,":",model.similar_by_vector(model[word], topn=10))
        #return dic
    #view_vectors(data_frame['con_title_content'][len(data_corpus["con_title_content"])-1],model)    
    #desc_word2vec[index]
    return json
    
    
QueryString="Among Deaths in 2016, a Heavy Toll in Pop Music - The New York Times"

final_result = word2vec_avg(QueryString)

print(final_result)    
    
    