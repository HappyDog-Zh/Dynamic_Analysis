#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud, STOPWORDS
import warnings
warnings.filterwarnings('ignore')


file_path = 'C:/Users/10474/Desktop/Info_vis/midterm/network_comment.csv'
df = pd.read_csv(file_path)

df = df.dropna(subset=['text'])

def get_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

df['sentiment'] = df['text'].apply(get_sentiment)

def classify_sentiment(polarity):
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

df['sentiment_category'] = df['sentiment'].apply(classify_sentiment)

sentiment_counts = df['sentiment_category'].value_counts()
plt.figure(figsize=(10, 6))
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='viridis')
plt.xlabel('Sentiment Category')
plt.ylabel('Number of Posts')
plt.title('Sentiment Analysis of Social Media Posts')
plt.show()


# In[2]:


df['creationDate'] = pd.to_datetime(df['creationDate'], errors='coerce')

df = df.dropna(subset=['creationDate'])

df['date'] = df['creationDate'].dt.date

daily_posts = df.groupby('date').size().reset_index(name='num_posts')

plt.figure(figsize=(12, 6))
plt.plot(daily_posts['date'], daily_posts['num_posts'], marker='o', linestyle='-')
plt.xlabel('Date')
plt.ylabel('Number of Posts')
plt.title('Trend of Social Media Posts Over Time')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# In[3]:


df = df.dropna(subset=['authorMeta/name'])
G = nx.Graph()

for _, row in df.iterrows():
    author = row['authorMeta/name']
    hashtags = row['input']
    if pd.notna(hashtags):
        hashtag_list = hashtags.split()  
        for hashtag in hashtag_list:
            G.add_edge(author, hashtag)

degree_threshold = 1
filtered_nodes = [node for node in G if G.degree(node) > degree_threshold]
G_filtered = G.subgraph(filtered_nodes)

pos = nx.spring_layout(G_filtered, k=0.3, seed=42)

edge_x = []
edge_y = []
for edge in G_filtered.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=0.5, color='#888'),
    hoverinfo='none',
    mode='lines'
)

node_x = []
node_y = []
for node in G_filtered.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers+text',
    text=[str(node) for node in G_filtered.nodes()],
    textposition='top center',
    hoverinfo='text',
    marker=dict(
        size=10,
        color='skyblue',
        line_width=2
    )
)

fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title='Filtered Network Analysis of Actors Involved in Conflicts',
                    titlefont=dict(size=16),
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20, l=5, r=5, t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                ))

fig.show()


# In[4]:


text_data = df['text'].values

vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
X = vectorizer.fit_transform(text_data)

num_topics = 5
lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
lda.fit(X)

words = np.array(vectorizer.get_feature_names_out())
topic_words = []
for topic_idx, topic in enumerate(lda.components_):
    top_words_idx = topic.argsort()[-10:][::-1]
    top_words = words[top_words_idx]
    topic_words.append(top_words)

topic_labels = [f'Topic {i+1}' for i in range(num_topics)]
word_labels = [', '.join(words) for words in topic_words]
word_counts = [topic.sum() for topic in lda.components_]

fig = px.scatter(
    x=topic_labels,
    y=[1] * len(topic_labels),
    size=word_counts,
    text=topic_labels,
    title='Topics Extracted from Social Media Posts',
    labels={'x': 'Topics', 'y': 'Frequency'},
    hover_data={'Words': word_labels},
    size_max=60
)
fig.update_traces(marker=dict(opacity=0.7, line=dict(width=2, color='DarkSlateGrey')))
fig.update_layout(
    xaxis={'visible': False},
    yaxis={'visible': False},
    showlegend=False
)

fig.show()


# In[ ]:





# In[13]:


import pandas as pd
import numpy as np
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import seaborn as sns

df = pd.read_csv('C:/Users/10474/Desktop/Info_vis/midterm/network_comment.csv')
df = df.dropna(subset=['text', 'creationDate', 'likesCount', 'commentsCount', 'viewsCount'])
df['creationDate'] = pd.to_datetime(df['creationDate'], errors='coerce')
df['date'] = df['creationDate'].dt.date

def get_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

def classify_sentiment(polarity):
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

df['sentiment'] = df['text'].apply(get_sentiment)
df['sentiment_category'] = df['sentiment'].apply(classify_sentiment)

sentiment_trends = df.groupby(['date', 'sentiment_category']).size().reset_index(name='count')

text_data = ' '.join(df['text'].values)
additional_stopwords = set(['and', 'of', 'in', 'to', 'for', 'on', 'with', 'at', 'by', 'from', 'is', 'that', 'it', 'as', 'an', 'this'])
stopwords = STOPWORDS.union(additional_stopwords)
wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis', max_words=200, stopwords=stopwords).generate(text_data)

engagement_metrics = df[['likesCount', 'commentsCount', 'viewsCount']]
corr_matrix = engagement_metrics.corr()
fig = plt.figure(figsize=(18, 12))

ax1 = fig.add_subplot(2, 1, 1)
sns.lineplot(
    data=sentiment_trends,
    x='date',
    y='count',
    hue='sentiment_category',
    palette='viridis',
    ax=ax1
)
ax1.set_title('Sentiment Trend Over Time', fontsize=16)
ax1.set_xlabel('Date', fontsize=12)
ax1.set_ylabel('Number of Posts', fontsize=12)
ax1.legend(title='Sentiment', loc='upper left', fontsize=10)
plt.xticks(rotation=45)

ax2 = fig.add_axes([0.05, 0.05, 0.4, 0.4])
ax2.imshow(wordcloud, interpolation='bilinear')
ax2.axis('off')
ax2.set_title('Word Cloud of Most Frequent Words', fontsize=16)

ax3 = fig.add_axes([0.55, 0.05, 0.4, 0.4])
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='viridis', ax=ax3, cbar_kws={'shrink': 0.8})
ax3.set_title('Correlation Heatmap of Engagement Metrics', fontsize=16)
ax3.set_xticklabels(corr_matrix.columns, rotation=45)
ax3.set_yticklabels(corr_matrix.columns, rotation=45)

plt.tight_layout(rect=[0, 0.4, 1, 1])
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




