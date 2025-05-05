import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from wordcloud import WordCloud, ImageColorGenerator, wordcloud
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize as wt

stop_words = set(stopwords.words('english'))

color_func1 = wordcloud.get_single_color_func('deepskyblue')

def word_extraction(sentence):
	words = wt(sentence)
	cleaned_text = [w for w in words if w not in stop_words]
	return cleaned_text


def create_cloud(source_text, mask_path = None):
	with open("./collections/"+source_text+'.txt', "r") as myfile:
		text = myfile.read().replace('Sir', '').replace('Lady', '')
		data = word_extraction(text)
		fdist = nltk.FreqDist(data)
		# known maps: summer, winter, grey
		colormap = 'summer'
		if mask_path:
			mask = np.array(Image.open('./masks/'+mask_path), np.int32)
			generated_wordcloud = WordCloud(collocations=False, width=7500, height=4400, mask=mask, min_font_size=14, max_words=800, background_color="white", contour_width=2, contour_color="black", colormap=colormap).fit_words(fdist)
		else:
			generated_wordcloud = WordCloud(collocations=False, width=7500, height=4400, min_font_size=14, max_words=800, background_color="white", contour_width=2, contour_color="black", colormap=colormap).fit_words(fdist)
		plt.figure(figsize=[75.0, 44.0])
		plt.imshow(generated_wordcloud, interpolation='bilinear')
		plt.axis("off")
		plt.savefig('./render/'+source_text+'.png')


create_cloud('experience')