import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import scipy.sparse as sps
from eals import ElementwiseAlternatingLeastSquares, load_model

from pathlib import Path

class eAlsPredictor:
    """
    A class to generate movie recommendations using the ElementwiseAlternatingLeastSquares model.

    Parameters:
    - model (ElementwiseAlternatingLeastSquares): A pre-trained model for generating user and item latent factors.
    - train_sparse (sps.csr_matrix): A sparse matrix representing the training data (user-item ratings).
    - movies (pd.DataFrame): A DataFrame containing information about movies (e.g., movie ID, title).
    """
    def __init__(self, model: ElementwiseAlternatingLeastSquares, train_sparse: sps.csr_matrix, movies: pd.DataFrame):
        """
        Initializes the eAlsPredictor class with a trained model, training data, and movie data.
        
        Parameters:
        - model (ElementwiseAlternatingLeastSquares): The trained ALS model used for making predictions.
        - train_sparse (sps.csr_matrix): A sparse matrix of training data.
        - movies (pd.DataFrame): DataFrame containing movie details (columns include 'movie_id', 'title').
        """
        self._model = model
        self._train_sparse = train_sparse
        self._movies = movies


    def _update_model(self, user_id: int, rated_items: dict[int, int]) -> None:
        """
        Updates the model based on new ratings provided by the user.
        
        Parameters:
        - user_id (int): The ID of the user.
        - rated_items (dict[int, int]): A dictionary containing item IDs and corresponding ratings to update the model.
        """
        for item_id, rating in rated_items.items():
            self._model.update_model(user_id, item_id, rating)


    def _get_recommend_items_ids(self, user_id: int, num_recommendations: int, num_random_items: int=0):
        """
        Returns recommended item IDs for a given user.
        
        Parameters:
        - user_id (int): The ID of the user for whom to generate recommendations.
        - num_recommendations (int): The number of top recommendations to generate.
        - num_random_items (int, optional): Number of random items to add to recommendations (default is 0).
        
        Returns:
        - np.ndarray: Array of recommended item IDs.
        """
        user_factors = self._model.user_factors
        item_factors = self._model.item_factors

        if user_id >= user_factors.shape[0]:
            raise ValueError(f"User ID {user_id} is out of range.")
        
        # Compute scores for all items for the given user
        user_vector = user_factors[user_id]
        scores = user_vector @ item_factors.T

        # Get the user's rated items from the training data
        user_rated_items = self._train_sparse[user_id].nonzero()[1]

        # Set scores for already rated items to a very low value to exclude them
        scores[user_rated_items] = -np.inf

        # Get the indices of the top N items with the highest scores
        top_items_indices = np.argsort(scores)[-num_recommendations:][::-1]

        # Add random items to the recommendations if num_random_items > 0
        if num_random_items > 0:
            random_items = np.random.choice(
                np.setdiff1d(np.arange(item_factors.shape[0]), top_items_indices),
                num_random_items,
                replace=False
            )
            top_items_indices = np.concatenate([top_items_indices, random_items])
        
        return top_items_indices


    def add_user(self, user_id: int, rated_items: dict[int, int]) -> None:
        new_user_ratings = np.zeros(self._train_sparse.shape[1], dtype=np.float32)

        # Update ratings for the rated items
        for item_id, rating in rated_items.items():
            new_user_ratings[item_id] = rating

        # Convert to a csr_matrix and stack with the existing training matrix
        new_user_sparse = sps.csr_matrix([new_user_ratings])
        self._train_sparse = sps.vstack([self._train_sparse, new_user_sparse])

        self._update_model(user_id, rated_items)


    def add_rating_for_user(self, user_id: int, item_id: int, rating: int) -> None:
        if user_id >= self._train_sparse.shape[0]:
            raise ValueError(f"User ID {user_id} is out of range. You may need to add the user first.")

        train_sparse_lil = self._train_sparse.tolil()
        train_sparse_lil[user_id, item_id] = rating

        self._train_sparse = train_sparse_lil.tocsr()
        self._update_model(user_id, {item_id: rating})


    def recommend_items(self, user_id: int, num_recommendations: int, num_random_items: int=0) -> pd.DataFrame:
        if user_id >= self._model.user_factors.shape[0]:
            raise ValueError(f"User ID {user_id} is out of range. You may need to add the user first.")
        recommended_items = self._get_recommend_items_ids(user_id, num_recommendations, num_random_items)
        
        return self._movies[self._movies['movie_id'].isin(recommended_items)]
        