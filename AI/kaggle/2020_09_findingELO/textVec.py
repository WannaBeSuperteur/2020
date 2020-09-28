from nltk.corpus import stopwords
from string import punctuation
from nltk.stem import WordNetLemmatizer # wnl
from nltk import pos_tag
from nltk import sent_tokenize, word_tokenize

# for method 2
# source: https://www.kaggle.com/alvations/basic-nlp-with-nltk
def penn2morphy(penntag):
    morphy_tag = {'NN':'n', 'JJ':'a',
                  'VB':'v', 'RB':'r'}
    try:
        return morphy_tag[penntag[:2]]
    except:
        return 'n' # if mapping isn't found, fall back to Noun.
    
def lemmatize_sent(text):
    wnl = WordNetLemmatizer()
    
    # Text input is string, returns lowercased strings.
    return [wnl.lemmatize(word.lower(), pos=penn2morphy(tag)) 
            for word, tag in pos_tag(word_tokenize(text))]

def preprocess_text(text):
    # Input: str, i.e. document/sentence
    # Output: list(str) , i.e. list of lemmas

    # In [26]:
    stopwords_json = {"en":["a","a's","able","about","above","according","accordingly","across","actually",
                            "after","afterwards","again","against","ain't","all","allow","allows","almost",
                            "alone","along","already","also","although","always","am","among","amongst","an",
                            "and","another","any","anybody","anyhow","anyone","anything","anyway","anyways",
                            "anywhere","apart","appear","appreciate","appropriate","are","aren't","around",
                            "as","aside","ask","asking","associated","at","available","away","awfully","b",
                            "be","became","because","become","becomes","becoming","been","before","beforehand",
                            "behind","being","believe","below","beside","besides","best","better","between",
                            "beyond","both","brief","but","by","c","c'mon","c's","came","can","can't","cannot",
                            "cant","cause","causes","certain","certainly","changes","clearly","co","com","come",
                            "comes","concerning","consequently","consider","considering","contain","containing",
                            "contains","corresponding","could","couldn't","course","currently","d","definitely",
                            "described","despite","did","didn't","different","do","does","doesn't","doing","don't",
                            "done","down","downwards","during","e","each","edu","eg","eight","either","else",
                            "elsewhere","enough","entirely","especially","et","etc","even","ever","every",
                            "everybody","everyone","everything","everywhere","ex","exactly","example","except",
                            "f","far","few","fifth","first","five","followed","following","follows","for",
                            "former","formerly","forth","four","from","further","furthermore","g","get","gets",
                            "getting","given","gives","go","goes","going","gone","got","gotten","greetings",
                            "h","had","hadn't","happens","hardly","has","hasn't","have","haven't","having",
                            "he","he's","hello","help","hence","her","here","here's","hereafter","hereby",
                            "herein","hereupon","hers","herself","hi","him","himself","his","hither","hopefully",
                            "how","howbeit","however","i","i'd","i'll","i'm","i've","ie","if","ignored",
                            "immediate","in","inasmuch","inc","indeed","indicate","indicated","indicates",
                            "inner","insofar","instead","into","inward","is","isn't","it","it'd","it'll","it's",
                            "its","itself","j","just","k","keep","keeps","kept","know","known","knows","l",
                            "last","lately","later","latter","latterly","least","less","lest","let","let's",
                            "like","liked","likely","little","look","looking","looks","ltd","m","mainly","many",
                            "may","maybe","me","mean","meanwhile","merely","might","more","moreover","most",
                            "mostly","much","must","my","myself","n","name","namely","nd","near","nearly",
                            "necessary","need","needs","neither","never","nevertheless","new","next","nine",
                            "no","nobody","non","none","noone","nor","normally","not","nothing","novel","now",
                            "nowhere","o","obviously","of","off","often","oh","ok","okay","old","on","once",
                            "one","ones","only","onto","or","other","others","otherwise","ought","our","ours",
                            "ourselves","out","outside","over","overall","own","p","particular","particularly",
                            "per","perhaps","placed","please","plus","possible","presumably","probably",
                            "provides","q","que","quite","qv","r","rather","rd","re","really","reasonably",
                            "regarding","regardless","regards","relatively","respectively","right","s","said",
                            "same","saw","say","saying","says","second","secondly","see","seeing","seem","seemed",
                            "seeming","seems","seen","self","selves","sensible","sent","serious","seriously",
                            "seven","several","shall","she","should","shouldn't","since","six","so","some",
                            "somebody","somehow","someone","something","sometime","sometimes","somewhat",
                            "somewhere","soon","sorry","specified","specify","specifying","still","sub","such",
                            "sup","sure","t","t's","take","taken","tell","tends","th","than","thank","thanks",
                            "thanx","that","that's","thats","the","their","theirs","them","themselves","then",
                            "thence","there","there's","thereafter","thereby","therefore","therein","theres",
                            "thereupon","these","they","they'd","they'll","they're","they've","think","third",
                            "this","thorough","thoroughly","those","though","three","through","throughout","thru",
                            "thus","to","together","too","took","toward","towards","tried","tries","truly","try"
                            ,"trying","twice","two","u","un","under","unfortunately","unless","unlikely","until",
                            "unto","up","upon","us","use","used","useful","uses","using","usually","uucp","v",
                            "value","various","very","via","viz","vs","w","want","wants","was","wasn't","way",
                            "we","we'd","we'll","we're","we've","welcome","well","went","were","weren't","what",
                            "what's","whatever","when","whence","whenever","where","where's","whereafter",
                            "whereas","whereby","wherein","whereupon","wherever","whether","which","while",
                            "whither","who","who's","whoever","whole","whom","whose","why","will","willing",
                            "wish","with","within","without","won't","wonder","would","wouldn't","x","y",
                            "yes","yet","you","you'd","you'll","you're","you've","your","yours","yourself",
                            "yourselves","z","zero"]}
    
    stopwords_json_en = set(stopwords_json['en'])
    stopwords_nltk_en = set(stopwords.words('english'))
    stopwords_punct = set(punctuation)
    stoplist_combined = set.union(stopwords_json_en, stopwords_nltk_en, stopwords_punct)
    
    return [word for word in lemmatize_sent(text) 
            if word not in stoplist_combined
            and not word.isdigit()]
