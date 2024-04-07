#importing required libraries
import pandas as pd
import gensim
import string
import nltk  #for word tokenization in pre-processing steps"
#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('stopwords')
from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords
stop = stopwords.words('english')

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

lemmatizer = nltk.stem.WordNetLemmatizer()

def doc2vec(Query_string):

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
#        stemming
        data_frame[column] = data_frame[column].apply(lambda x:' '.join([stemmer.stem(y) 
                                                                                       for y in word_tokenize(str(x))]))
#        lemmatization
        data_frame[column] = data_frame[column].apply(lambda x:' '.join([lemmatizer.lemmatize(w) 
                                                                                        for w in word_tokenize(str(x))]))
        return data_frame[column]
    
    data_frame['con_title_content'] = preprocessing(data_frame, 'con_title_content')
    
    #The below statement is for exporting the Pre-Processed Corpus to Excel Sheeet.
    #export_excel = data_frame.to_excel (r'C:\Users\ksingara\Documents\POC_NLP_Similarity\Article_Matching\export_dataframe.xlsx', index = None, header=True) #Note to add '.xlsx' at the end of the path
    
    #Storing the documents of the corpus in a list.
    list_desc = data_frame["con_title_content"].tolist()
    
    
    #preparing data for Doc2vec training
    tokens = []
    for sentence in list_desc:
        tokens.append(word_tokenize(sentence))
    
    
    train_corpus = []
    for i, desc in enumerate(tokens):
        train_corpus.append(gensim.models.doc2vec.TaggedDocument(desc, [i]))
    #trrain doc2vec model    
    model = gensim.models.doc2vec.Doc2Vec(vector_size=100, min_count=2, epochs=40)
    model.build_vocab(train_corpus)
       
    desc_doc2vec = model.docvecs
    #print(desc_doc2vec)
    #Function for finding the cosine similarities and getting top 10 documents with respect to their similarity scores.
    # In D2V, since the matrix size is big, hence we shall be using the spatial distance cosine similarity
    def similar_desc_indexes(desc_doc2vec,query_index):
        temp_retrieval = model.docvecs.most_similar([model.docvecs[-1]], topn=len(model.docvecs))
        temp_retrieval = sorted(temp_retrieval)[0:len(temp_retrieval)-2]
        retrieval = [i[1] for i in temp_retrieval]
        #print(retrieval)
        #for i in range(len(desc_doc2vec)-2):
         #   retrieval.append(1 - spatial.distance.cosine(desc_doc2vec[query_index], desc_doc2vec[i]))
        #retrive indices of top 5 similar descriptions
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
    QuryStrresult = similar_desc_indexes(desc_doc2vec,query_index = -1)
    result = QuryStrresult
    #Calling the function for getting top5 results out of Query String
    article_list = get_top_descriptions(result)

    #Converting the List to JSON sequential Objects.
    json=article_list[0:5].to_json(orient='index')
        
    return json
    
    
QueryString="Among Deaths in 2016, a Heavy Toll in Pop Music - The New York Times"

final_result = doc2vec(QueryString)

print(final_result)    
    
    