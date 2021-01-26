import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from wordcloud import WordCloud
from tkinter import *


df = pd.read_csv('spam.csv',encoding='latin-1')
df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis = 1)

df.rename(columns = {'v1':'labels', 'v2':'message'}, inplace=True)
df.drop_duplicates(inplace=True)
df['label'] = df['labels'].map({'ham': 0, 'spam': 1})
df.drop(['labels'], axis=1, inplace=True)


import string


def preprocess_text(message):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text

    """
    # Check characters to see if they are in punctuation
    without_punc = [char for char in message if char not in string.punctuation]

    # Join the characters again to form the string.
    without_punc = ''.join(without_punc)

    # Now just remove any stopwords
    return [word for word in without_punc.split() if word.lower() not in stopwords.words('english')]

df['message'].head().apply(preprocess_text)



# spam_words = ' '.join(list(df[df['label'] == 1]['message']))
# spam_wc = WordCloud(width = 512,height = 512).generate(spam_words)
# plt.figure(figsize = (10, 8), facecolor = 'k')
# plt.imshow(spam_wc)
# plt.show()
#
# from wordcloud import WordCloud
#
# ham_words = ' '.join(list(df[df['label'] == 0]['message']))
# ham_wc = WordCloud(width = 512,height = 512).generate(ham_words)
# plt.figure(figsize = (10, 8), facecolor = 'k')
# plt.imshow(ham_wc)
# plt.show()


from sklearn.feature_extraction.text import CountVectorizer
x = df['message']
y = df['label']
cv = CountVectorizer()
x= cv.fit_transform(x)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)


from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB().fit(x_train, y_train)


# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# pred = classifier.predict(x_test)
# print(classification_report(y_test, pred))
# print()
# print('Confusion Matrix:\n',confusion_matrix(y_test, pred))
# print()
# print('Accuracy : ',accuracy_score(y_test, pred))
#
#
# # print the predictions
# print(classifier.predict(x_test))
#
# # print the actual values
# print(y_test.values)


def sms():
    # creating a list of labels
    lab = ['not spam', 'spam']

    # perform tokenization
    x = cv.transform([e.get()]).toarray()

    # predict the text
    p = classifier.predict(x)

    # convert the words in string with the help of list
    s = [str(i) for i in p]
    a = int("".join(s))

    # show out the final result
    res = str("This is " + lab[a])

    if lab[a]=='spam':

        classification = Label(root, text=res, font=('helvetica', 15 , 'bold'), fg="red")
        classification.pack()
    else:
        classification = Label(root, text=res, font=('helvetica', 15, 'bold'), fg="green")
        classification.pack()




root = Tk()
root.title('SpellCheck')
root.geometry('400x400')

head = Label(root, text='SPAM  Checker',font=('helvetica', 24 , 'bold'))
head.pack()
e = Entry(root, width=400,borderwidth=5)
e.pack()
b = Button(root, text = 'Check', font=('helvetica', 20 , 'bold'), fg = 'white', bg = 'green', command = sms)
b.pack()
root.mainloop()

