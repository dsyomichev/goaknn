import nltk  # type: ignore

from nltk.corpus import stopwords  # type: ignore
from nltk.downloader import Downloader  # type: ignore
from nltk.stem import PorterStemmer  # type: ignore

downloader = Downloader()
stemmer = PorterStemmer()

if downloader.status("stopwords") != Downloader.INSTALLED:  # type: ignore
    nltk.download("stopwords")  # type: ignore

stop_words: list[str] = [stemmer.stem(word) for word in stopwords.words("english")]  # type: ignore
