import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import numpy as np

# Etapa 1: Leitura do arquivo u.item
# Nota: O encoding 'ISO-8859-1' é usado pois o arquivo u.item contém caracteres especiais
movies_df = pd.read_csv('data/u.item', sep='|', encoding='ISO-8859-1',
                        names=['movieId', 'title', 'release date', 'video release date',
                               'IMDb URL', 'unknown', 'Action', 'Adventure', 'Animation',
                               'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                               'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                               'Thriller', 'War', 'Western'],
                        usecols=['movieId', 'title', 'Action', 'Adventure', 'Animation',
                                 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                                 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                                 'Thriller', 'War', 'Western'])
# Etapa 2: Preparação dos Dados
# Concatenar os gêneros para cada filme
genres = ['Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary',
          'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
          'Thriller', 'War', 'Western']


# Criar uma string única de gêneros para cada filme
for i in range(movies_df.shape[0]):
  genres_list = []
  for genre in genres:
    if movies_df.loc[i, genre] == 1:
      genres_list.append(genre)
  movies_df.loc[i, 'features'] = ' '.join(genres_list)


# Etapa 3: Sistema de Recomendação
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_df['features'])

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


def recommend_movies(movie_title, cosine_sim=cosine_sim):
    # Encontrar o índice do filme dado seu título
    idx = movies_df.index[movies_df['title'] == movie_title].tolist()[0]

    # Obter as pontuações de similaridade de todos os filmes com esse filme
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Ordenar os filmes com base nas pontuações de similaridade
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Obter os scores dos 10 filmes mais similares
    sim_scores = sim_scores[1:11]

    # Obter os índices dos filmes
    movie_indices = [i[0] for i in sim_scores]

    # Retornar os títulos dos melhores filmes
    return movies_df['title'].iloc[movie_indices]


# Teste da recomendação
recommended_movies = recommend_movies('Toy Story (1995)')
print("Movies Recommended:")
print(recommended_movies)
