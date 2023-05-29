from fastapi import FastAPI
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import csv
from sqlalchemy import create_engine


app = FastAPI()
recommendations = dict()

def write_2d_array_to_csv(data, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)

@app.get("/train")
def train():
    db_string = 'postgresql+psycopg2'+"://"+'sakib1'+":" + \
            'sakib123' + "@"+'localhost:5432'+'/'+'geekbangla_dev'
    db = create_engine(db_string)
    query = 'select * from \"core\".user_category'
    df = pd.read_sql_query(query, con=db)
    pivoteDF = df.pivot_table(
        index='userId', columns='categoryId', values="score", aggfunc='sum')
    pivoteDF
    dataset = pivoteDF.values
    print("dataset\n", dataset)

    # Calculate the user similarity matrix
    user_similarity = cosine_similarity(dataset)
    write_2d_array_to_csv(user_similarity, "user_sim.csv")
    for user in range(dataset.shape[0]):
        recommendation = getRecommendation(user, user_similarity, dataset)
        recommendations[user] = recommendation
    return {"message" : "User Similarity Calculated, User recommendations calculated"}

@app.get("/recommend/{user_id}")
def read_item(user_id: int):
    return {"recommendations": recommendations[user_id]}



def getRecommendation(target_user, user_similarity, dataset):
     # Find the most similar users to the target user
    similar_users = np.argsort(user_similarity[target_user])[::-1][1:]
    similar_users = similar_users[:3]

    # Recommend categories based on similar users' preferences
    recommendation = []
    for category in range(dataset.shape[1]):
        if dataset[target_user, category] == 0:  # User has not rated this category
            category_rating = 0
            similarity_sum = 0
            for user in similar_users:
                if dataset[user, category] != 0:  # Similar user has rated this category
                    category_rating += dataset[user, category] * \
                        user_similarity[target_user, user]
                    similarity_sum += user_similarity[target_user, user]
            if similarity_sum > 0:
                category_rating /= similarity_sum
            recommendation.append((category+1, category_rating))
    
    # Sort the recommendations by rating (descending order)
    recommendation = sorted(
        recommendation, key=lambda x: x[1], reverse=True)
    # Print the recommendations
    print("Recommendations for User", target_user + 1)
    return recommendation
