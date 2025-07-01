import requests

url = "http://localhost:8000/infer"
payload = {
    "instruction": "Given a web search query, retrieve relevant passages that answer the query",
    "pairs": [
        {
            "query": "What is the capital of China?",
            "document": "The capital of China is Beijing."
        },
        {
            "query": "Explain gravity",
            "document": "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun."
        }
    ]
}

response = requests.post(url, json=payload)
response.raise_for_status()
print("scores:", response.json()["scores"])