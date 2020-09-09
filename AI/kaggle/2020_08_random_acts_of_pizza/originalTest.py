# source: https://www.kaggle.com/alvations/basic-nlp-with-nltk/
import nltk
from nltk.corpus import brown
from nltk.corpus import webtext
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from collections import Counter
from io import StringIO
from sklearn.feature_extraction.text import CountVectorizer
from operator import itemgetter
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

### FUNCTIONS ###

def penn2morphy(penntag):
    """ Converts Penn Treebank tags to WordNet. """
    morphy_tag = {'NN':'n', 'JJ':'a',
                  'VB':'v', 'RB':'r'}
    try:
        return morphy_tag[penntag[:2]]
    except:
        return 'n' # if mapping isn't found, fall back to Noun. 
    
def lemmatize_sent(text): 
    # Text input is string, returns lowercased strings.
    return [wnl.lemmatize(word.lower(), pos=penn2morphy(tag)) 
            for word, tag in pos_tag(word_tokenize(text))]
    
# `pos_tag` takes the tokenized sentence as input, i.e. list of string,
# and returns a tuple of (word, tg), i.e. list of tuples of strings
# so we need to get the tag from the 2nd element.

def preprocess_text(text):
    # Input: str, i.e. document/sentence
    # Output: list(str) , i.e. list of lemmas
    return [word for word in lemmatize_sent(text) 
            if word not in stoplist_combined
            and not word.isdigit()]

### MAIN CODE ###

if __name__ == '__main__':
    
    # Stopwords from stopwords-json
    stopwords_json = {"en":["a","a's","able","about","above","according","accordingly",
                            "across","actually","after","afterwards","again","against",
                            "ain't","all","allow","allows","almost","alone","along","already",
                            "also","although","always","am","among","amongst","an","and",
                            "another","any","anybody","anyhow","anyone","anything","anyway",
                            "anyways","anywhere","apart","appear","appreciate","appropriate",
                            "are","aren't","around","as","aside","ask","asking","associated",
                            "at","available","away","awfully","b","be","became","because",
                            "become","becomes","becoming","been","before","beforehand","behind",
                            "being","believe","below","beside","besides","best","better",
                            "between","beyond","both","brief","but","by","c","c'mon","c's",
                            "came","can","can't","cannot","cant","cause","causes","certain",
                            "certainly","changes","clearly","co","com","come","comes",
                            "concerning","consequently","consider","considering","contain",
                            "containing","contains","corresponding","could","couldn't",
                            "course","currently","d","definitely","described","despite",
                            "did","didn't","different","do","does","doesn't","doing","don't",
                            "done","down","downwards","during","e","each","edu","eg","eight",
                            "either","else","elsewhere","enough","entirely","especially","et",
                            "etc","even","ever","every","everybody","everyone","everything",
                            "everywhere","ex","exactly","example","except","f","far","few",
                            "fifth","first","five","followed","following","follows","for",
                            "former","formerly","forth","four","from","further","furthermore",
                            "g","get","gets","getting","given","gives","go","goes","going",
                            "gone","got","gotten","greetings","h","had","hadn't","happens",
                            "hardly","has","hasn't","have","haven't","having","he","he's",
                            "hello","help","hence","her","here","here's","hereafter","hereby",
                            "herein","hereupon","hers","herself","hi","him","himself","his",
                            "hither","hopefully","how","howbeit","however","i","i'd","i'll",
                            "i'm","i've","ie","if","ignored","immediate","in","inasmuch","inc",
                            "indeed","indicate","indicated","indicates","inner","insofar",
                            "instead","into","inward","is","isn't","it","it'd","it'll","it's",
                            "its","itself","j","just","k","keep","keeps","kept","know","known",
                            "knows","l","last","lately","later","latter","latterly","least",
                            "less","lest","let","let's","like","liked","likely","little","look",
                            "looking","looks","ltd","m","mainly","many","may","maybe","me",
                            "mean","meanwhile","merely","might","more","moreover","most",
                            "mostly","much","must","my","myself","n","name","namely","nd","near",
                            "nearly","necessary","need","needs","neither","never","nevertheless",
                            "new","next","nine","no","nobody","non","none","noone","nor",
                            "normally","not","nothing","novel","now","nowhere","o","obviously",
                            "of","off","often","oh","ok","okay","old","on","once","one",
                            "ones","only","onto","or","other","others","otherwise","ought",
                            "our","ours","ourselves","out","outside","over","overall","own","p",
                            "particular","particularly","per","perhaps","placed","please",
                            "plus","possible","presumably","probably","provides","q","que",
                            "quite","qv","r","rather","rd","re","really","reasonably","regarding",
                            "regardless","regards","relatively","respectively","right","s",
                            "said","same","saw","say","saying","says","second","secondly","see",
                            "seeing","seem","seemed","seeming","seems","seen","self","selves",
                            "sensible","sent","serious","seriously","seven","several","shall",
                            "she","should","shouldn't","since","six","so","some","somebody",
                            "somehow","someone","something","sometime","sometimes","somewhat",
                            "somewhere","soon","sorry","specified","specify","specifying","still",
                            "sub","such","sup","sure","t","t's","take","taken","tell","tends",
                            "th","than","thank","thanks","thanx","that","that's","thats","the",
                            "their","theirs","them","themselves","then","thence","there",
                            "there's","thereafter","thereby","therefore","therein","theres",
                            "thereupon","these","they","they'd","they'll","they're","they've",
                            "think","third","this","thorough","thoroughly","those","though",
                            "three","through","throughout","thru","thus","to","together","too",
                            "took","toward","towards","tried","tries","truly","try","trying",
                            "twice","two","u","un","under","unfortunately","unless","unlikely",
                            "until","unto","up","upon","us","use","used","useful","uses","using",
                            "usually","uucp","v","value","various","very","via","viz","vs","w",
                            "want","wants","was","wasn't","way","we","we'd","we'll","we're",
                            "we've","welcome","well","went","were","weren't","what","what's",
                            "whatever","when","whence","whenever","where","where's","whereafter",
                            "whereas","whereby","wherein","whereupon","wherever","whether",
                            "which","while","whither","who","who's","whoever","whole","whom",
                            "whose","why","will","willing","wish","with","within","without",
                            "won't","wonder","would","wouldn't","x","y","yes","yet","you","you'd",
                            "you'll","you're","you've","your","yours","yourself","yourselves",
                            "z","zero"]}

    stopwords_json_en = set(stopwords_json['en'])
    stopwords_nltk_en = set(stopwords.words('english'))
    stopwords_punct = set(punctuation)

    # Combine the stopwords. Its a lot longer so I'm not printing it out...
    stoplist_combined = set.union(stopwords_json_en, stopwords_nltk_en, stopwords_punct)

    porter = PorterStemmer()
    wnl = WordNetLemmatizer()

    with open('train.json') as fin:
        trainjson = json.load(fin)

    with open('test.json') as fin:
        testjson = json.load(fin)

    df = pd.io.json.json_normalize(trainjson) # Pandas magic... 
    df_train = df[['request_id', 'request_title', 
                   'request_text_edit_aware', 
                   'requester_received_pizza']]

    df = pd.io.json.json_normalize(testjson) # Pandas magic... 
    df_test = df[['request_id', 'request_title', 
                   'request_text_edit_aware']]

    print(' <<< 0. df_train >>>')
    print(df_train)

    print('\n\n <<< 1. df_test >>>')
    print(df_test)

    # It doesn't really matter what the function name is called
    # but the `train_test_split` is splitting up the data into 
    # 2 parts according to the `test_size` argument you've set.

    # When we're splitting up the training data, we're spltting up 
    # into train, valid split. The function name is just a name =)
    train, valid = train_test_split(df_train, test_size=0.2)

    print('\n\n <<< 2. train >>>')
    print(train)

    print('\n\n <<< 3. valid >>>')
    print(valid)

    # Initialize the vectorizer and 
    # override the analyzer totally with the preprocess_text().
    # Note: the vectorizer is just an 'empty' object now.
    count_vect = CountVectorizer(analyzer=preprocess_text)

    print('\n\n <<< 4. count_vect >>>')
    print(count_vect)

    # When we use `CounterVectorizer.fit_transform`,
    # we essentially create the dictionary and 
    # vectorize our input text at the same time.
    train_set = count_vect.fit_transform(train['request_text_edit_aware'])
    train_tags = train['requester_received_pizza']

    # When vectorizing the validation data, we use `CountVectorizer.transform()`.
    valid_set = count_vect.transform(valid['request_text_edit_aware'])
    valid_tags = valid['requester_received_pizza']

    # When vectorizing the test data, we use `CountVectorizer.transform()`.
    test_set = count_vect.transform(df_test['request_text_edit_aware'])

    print('\n\n <<< 5. train_set >>>')
    print(train_set)

    print('\n\n <<< 6. train_tags >>>')
    print(train_tags)

    print('\n\n <<< 7. valid_set >>>')
    print(valid_set)

    print('\n\n <<< 8. valid_tags >>>')
    print(valid_tags)

    print('\n\n <<< 9. count_vect >>>')
    print(test_set)

    clf = MultinomialNB()

    print('\n\n <<< 10. clf >>>')
    print(clf)

    # To train the classifier, simple do 
    clf.fit(train_set, train_tags) 

    # To predict our tags (i.e. whether requesters get their pizza), 
    # we feed the vectorized `test_set` to .predict()
    predictions_valid = clf.predict(valid_set)

    print('\n\n <<< 11. predictions_valid >>>')
    print(predictions_valid)

    print('Pizza reception accuracy = {}'.format(
            accuracy_score(predictions_valid, valid_tags) * 100)
         )

    count_vect = CountVectorizer(analyzer=preprocess_text)

    print('\n\n <<< 12. count_vect >>>')
    print(count_vect)

    full_train_set = count_vect.fit_transform(df_train['request_text_edit_aware'])
    full_tags = df_train['requester_received_pizza']

    print('\n\n <<< 13. full_train_set >>>')
    print(full_train_set)
    
    print('\n\n <<< 14. full_tags >>>')
    print(full_tags)

    # Note: We have to re-vectorize the test set since
    #       now our vectorizer is different using the full 
    #       training set.
    test_set = count_vect.transform(df_test['request_text_edit_aware'])

    print('\n\n <<< 15. test_set >>>')
    print(test_set)

    # To train the classifier
    clf = MultinomialNB()

    print('\n\n <<< 16. clf >>>')
    print(clf)

    clf.fit(full_train_set, full_tags)

    # To predict our tags (i.e. whether requesters get their pizza), 
    # we feed the vectorized `test_set` to .predict()
    predictions = clf.predict(test_set)

    # PRINT RESULT
    print(predictions)
