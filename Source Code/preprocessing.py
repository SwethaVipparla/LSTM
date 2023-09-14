
import nltk
import unicodedata
import re

with open('Auguste_Maquet.txt', 'r') as f:
    text = f.read()

def normalize_unicode(s):
    return unicodedata.normalize('NFD', s)

def preprocess_text(text):
    text = normalize_unicode(text)
    text = re.sub(r"(.)(\1{2,})", r"\1", text)
    text = re.sub(r"[^a-zA-Z.,!?]+", " ", text)
    text = text.strip().lower()
    return text

nltk.download('punkt')

sentences = nltk.sent_tokenize(preprocess_text(text))
sentences = [sent for sent in sentences if not sent.startswith('chapter')]
sentences = [nltk.word_tokenize(sent) for sent in sentences]
sentences = [['sos'] + sent + ['eos'] for sent in sentences]