import pytest
import numpy as np
import pandas as pd
import scipy.sparse as sps
from unittest.mock import MagicMock

from model.model import eAlsPredictor  

@pytest.fixture
def mock_model():
    """Fixture for a mock ElementwiseAlternatingLeastSquares model."""
    model = MagicMock()
    model.user_factors = np.array([
        [0.1, 0.2],
        [0.3, 0.4],
        [0.5, 0.6]
    ])
    model.item_factors = np.array([
        [0.1, 0.2],
        [0.3, 0.4],
        [0.5, 0.6],
        [0.7, 0.8]
    ])
    return model

@pytest.fixture
def train_sparse():
    """Fixture for a sample sparse matrix representing user-item ratings."""
    return sps.csr_matrix([
        [5, 0, 0, 1],
        [0, 3, 0, 0],
        [0, 0, 4, 0]
    ])

@pytest.fixture
def movies():
    """Fixture for a sample movies DataFrame."""
    return pd.DataFrame({
        'movie_id': [0, 1, 2, 3],
        'title': ["Movie A", "Movie B", "Movie C", "Movie D"]
    })

@pytest.fixture
def predictor(mock_model, train_sparse, movies):
    """Fixture for the eAlsPredictor instance."""
    return eAlsPredictor(mock_model, train_sparse, movies)


def test_recommend_items(predictor):
    """Test the recommend_items method."""
    recommendations = predictor.recommend_items(user_id=0, num_recommendations=2)
    assert len(recommendations) == 2
    assert set(recommendations['movie_id']).issubset({0, 1, 2, 3})


def test_add_user(predictor):
    """Test the add_user method."""
    user_id = 3
    rated_items = {1: 5, 2: 4}
    predictor.add_user(user_id, rated_items)

    assert predictor._train_sparse.shape[0] == 4  # New user added
    assert predictor._train_sparse[3, 1] == 5
    assert predictor._train_sparse[3, 2] == 4
    predictor._model.update_model.assert_called()


def test_add_rating_for_user(predictor):
    """Test the add_rating_for_user method."""
    user_id = 0
    item_id = 2
    rating = 5
    predictor.add_rating_for_user(user_id, item_id, rating)

    assert predictor._train_sparse[0, 2] == 5
    predictor._model.update_model.assert_called_with(user_id, item_id, rating)


def test_recommend_items_with_random(predictor):
    """Test the recommend_items method with random recommendations."""
    recommendations = predictor.recommend_items(user_id=1, num_recommendations=2, num_random_items=1)
    assert len(recommendations) == 3


def test_get_recommend_items_ids_invalid_user(predictor):
    """Test _get_recommend_items_ids for invalid user ID."""
    with pytest.raises(ValueError):
        predictor._get_recommend_items_ids(user_id=10, num_recommendations=2)


def test_add_rating_for_invalid_user(predictor):
    """Test add_rating_for_user with an invalid user ID."""
    with pytest.raises(ValueError):
        predictor.add_rating_for_user(user_id=10, item_id=2, rating=5)
