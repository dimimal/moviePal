import pandas as pd
import difflib
import string
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

###function to clear our inputs 
def clearSent(sentence):
    
    stop_words = set(stopwords.words("english"))
    stop_words.update(['.', ',', '"', "'", '``', '&', '?', '!', ':', ';', '(', ')', '[', ']', '{', '}','see','watch','movie'] )

    
    sentence = sentence.lower()
    words_sent = word_tokenize(sentence)
    filtered_sent = [w for w in words_sent if not w in stop_words]
    
    new_sent = "".join(["|"+i if not i.startswith("'") and i not in string.punctuation else i for i in filtered_sent]).strip()
    
    return(new_sent)
	
	

#### first interaction with user

featvec= []
genre = input("Hey buddy, tell me what genres are you looking for? You can type 1 or more")
if ((genre == "")|(genre == " ")) : featvec.append(-1)

else: featvec.append(clearSent(str(genre)))

featvec

#### bubble 1 - actors

actors = input("What actor/s would you like to play?")
if ((actors == "")|(actors == " ")) : featvec.append(-1)

else: featvec.append(clearSent(str(actors)))   

featvec


#### bubble 2 - imdb_rating 

rating = input("Please, type a score of your choice...")
if ((rating == "")|(rating == " ")) : featvec.append(-1)

else: featvec.append(clearSent(str(rating)))   

featvec


### bubble 3 - year

year = input("Give me the release date...")
if ((year == "")|(year == " ")) : featvec.append(-1)

else: featvec.append(clearSent(str(year)))   

featvec


### bubble 4 - similar movies
featvec = [-1 for i in range(5)]

sim_movies = input("Type the title of a movie and i will recomend you similar")
if ((sim_movies == "")|(sim_movies == " ")) : featvec.append(-1)

else: featvec[4] = clearSent(str(sim_movies)) #featvec.append(clearSent(str(sim_movies)))   

print(featvec)

data = pd.read_csv("movie_metadata.csv")



xxx = pd.DataFrame({"title": data["movie_title"].values, 
                               "genres": data["genres"].values})