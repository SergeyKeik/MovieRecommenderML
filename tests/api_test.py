import pytest
import pandas as pd
from unittest.mock import patch
from fastapi.testclient import TestClient
from src.main import app  


@pytest.fixture
def client():
    """Provide a FastAPI test client."""
    return TestClient(app)

@patch("src.main.recommender")
@patch("src.main.movie_id_mapping", {1: 10, 2: 20})
def test_create_user(mock_recommender, client):
    """Test the PUT /api/users/{user_id} endpoint."""
    mock_recommender.add_user.return_value = None

    response = client.put(
        "/api/users/6040",
        json={"rates": [{"movie_id": 1, "rate": 5}, {"movie_id": 2, "rate": 4}]},
    )

    assert response.status_code == 200
    assert response.json() == {"message": "User 6040 added successfully."}
    mock_recommender.add_user.assert_called_once_with(user_id=6040, rated_items={10: 5, 20: 4})

@patch("src.main.recommender")
@patch("src.main.movie_id_mapping", {1: 10, 2: 20})
def test_update_user(mock_recommender, client):
    """Test the PATCH /api/users/{user_id}/rates endpoint."""
    mock_recommender.add_rating_for_user.return_value = None

    response = client.patch(
        "/api/users/1/rates",
        json={"rates": [{"movie_id": 1, "rate": 5}, {"movie_id": 2, "rate": 3}]},
    )

    assert response.status_code == 200
    assert response.json() == {"message": "User 1's ratings updated successfully."}
    mock_recommender.add_rating_for_user.assert_any_call(user_id=0, item_id=10, rating=5)
    mock_recommender.add_rating_for_user.assert_any_call(user_id=0, item_id=20, rating=3)

@patch("src.main.recommender")
def test_get_recommendations(mock_recommender, client):
    """Test the GET /api/recomms/{user_id} endpoint."""
    mock_recommender.recommend_items.return_value = pd.DataFrame.from_dict([
        {"movie_id": 1, "title": "Mocked Movie A"},
        {"movie_id": 2, "title": "Mocked Movie B"},
    ])

    response = client.get("/api/recomms/1?size=2")

    assert response.status_code == 200
    assert response.json() == {"movie_ids": [1, 2]}
    mock_recommender.recommend_items.assert_called_once_with(user_id=0, num_recommendations=2)

@patch("src.main.recommender")
def test_get_recommendations_invalid_user(mock_recommender, client):
    """Test the GET /api/recomms/{user_id} endpoint with an invalid user."""
    mock_recommender.recommend_items.side_effect = ValueError("User ID 998 is out of range.")

    response = client.get("/api/recomms/999")

    assert response.status_code == 400
    assert response.json() == {"detail": "User ID 998 is out of range."}
    mock_recommender.recommend_items.assert_called_once_with(user_id=998, num_recommendations=5)
