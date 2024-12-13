from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import sys
from pathlib import Path
import scipy.sparse as sps
from eals import load_model

# sys.path.append(str(Path(__file__).resolve().parent.parent))

from pathlib import Path
from model.model import eAlsPredictor

app = FastAPI()

# Paths to run in docker
MODEL_PATH = Path("/app/data/model.joblib")
MOVIES_PATH = Path("/app/data/movies.csv")
TRAIN_DATA_PATH = Path("/app/data/ratings.npz")

# Paths to run locally
# MODEL_PATH = Path("model.joblib")
# MOVIES_PATH = Path("movies.csv")
# TRAIN_DATA_PATH = Path("ratings.npz")

ratings = pd.read_csv('ratings.csv', sep='\t', encoding='latin-1', usecols=['user_id', 'movie_id', 'rating', 'timestamp'])
unique_movie_ids = ratings['movie_id'].unique()

movie_id_mapping = {movie_id: idx for idx, movie_id in enumerate(unique_movie_ids)}

model = load_model(MODEL_PATH)
movies = pd.read_csv(MOVIES_PATH, sep='\t', encoding='latin-1', usecols=['movie_id', 'title', 'genres'])
train_sparse = sps.load_npz(TRAIN_DATA_PATH)

recommender = eAlsPredictor(model=model, train_sparse=train_sparse, movies=movies)

class UserMovieRateDto(BaseModel):
    movie_id: int
    rate: int


class CreateUserRequestDto(BaseModel):
    rates: List[UserMovieRateDto]


class UpdateUserRequestDto(BaseModel):
    rates: List[UserMovieRateDto]


class RecommsResponseDto(BaseModel):
    movie_ids: List[int]


@app.put("/api/users/{user_id}")
async def create_user(user_id: int, request: CreateUserRequestDto):
    try:
        ratings = {rate.movie_id: rate.rate for rate in request.rates}
        ratings = {movie_id_mapping[item_id]: rate for item_id, rate in ratings.items()}
        recommender.add_user(user_id=user_id, rated_items=ratings)
        return {"message": f"User {user_id} added successfully."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.patch("/api/users/{user_id}/rates")
async def update_user(user_id: int, request: UpdateUserRequestDto):
    try:
        for rate in request.rates:
            recommender.add_rating_for_user(user_id=user_id-1, item_id=rate.movie_id, rating=rate.rate)
        return {"message": f"User {user_id}'s ratings updated successfully."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/recomms/{user_id}")
async def get_recommendations(
    user_id: int, size: Optional[int] = Query(None, alias="size", description="Number of recommendations")
):
    try:
        num_recommendations = size if size is not None else 5  
        recommendations = recommender.recommend_items(user_id=user_id-1, num_recommendations=num_recommendations)
        movie_ids = recommendations["movie_id"].tolist()
        return RecommsResponseDto(movie_ids=movie_ids)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
