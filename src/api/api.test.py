import requests
import json


def test_api_simple():
    base_url = "http://localhost:8001"

    try:
        # Test health
        response = requests.get(f"{base_url}/health")
        print("Health Check:", response.json())

        # Test prediction
        response = requests.post(
            f"{base_url}/predict", json={"text": "هذا المنتج رائع جداً 😍"}
        )
        result = response.json()
        print("\nPrediction Result:")
        print(json.dumps(result, indent=2, ensure_ascii=False))

        # Batch prediction
        response = requests.post(
            f"{base_url}/predict/batch",
            json={"text": ["منتج جميل", "لا يعمل", "المنتج سيء جداً 😡"]},
        )
        result = response.json()
        print("\nBatch Prediction Result:")
        print(json.dumps(result, indent=2, ensure_ascii=False))

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    test_api_simple()
