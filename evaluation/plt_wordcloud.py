from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from indices_pt_repr import PainDataset

pos_dataset = PainDataset(pos=True)
dictionary = pos_dataset.dictionary
words = list(dictionary.token2id.keys())

def create_word_cloud(string):
   maskArray = np.array(Image.open("cloud.png"))
   cloud = WordCloud(background_color = "white", max_words = 200, mask = maskArray, stopwords = set(STOPWORDS))
   cloud.generate(string)
   cloud.to_file("wordCloud.png")


dataset = ' '.join(x for x in words)

stopwords = set(STOPWORDS)

wordcloud = WordCloud(width=800, height=800,
                      background_color='white',
                      stopwords=stopwords,
                      min_font_size=10).generate(dataset)

# plot the WordCloud image
plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.savefig('wordcloud.png')
plt.show()
