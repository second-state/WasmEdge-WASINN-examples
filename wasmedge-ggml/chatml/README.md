# `chatml`

## Execute

````console
$ wasmedge --dir .:. \
  --nn-preload default:GGML:AUTO:mixtral-8x7b-instruct-v0.1.Q5_K_M.gguf \
  wasmedge-ggml-chatml.wasm default

USER:
List 3 most popular books. The information should include their title, author, published year, and genre. Format the answer into JSON objects.
ASSISTANT:
[
  {
    "title": "To Kill a Mockingbird",
    "author": "Harper Lee",
    "published_year": 1960,
    "genre": "Fiction"
  },
  {
    "title": "The Great Gatsby",
    "author": "F. Scott Fitzgerald",
    "published_year": 1925,
    "genre": "Fiction"
  },
  {
    "title": "Pride and Prejudice",
    "author": "Jane Austen",
    "published_year": 1813,
    "genre": "Romance"
  }
]
USER:
Implement a HTTP API service in python to show these results.
ASSISTANT:
Here is a simple implementation using Flask, a Python web framework:

```python
from flask import Flask, jsonify

app = Flask(__name__)

books = [
  {
    "title": "To Kill a Mockingbird",
    "author": "Harper Lee",
    "published_year": 1960,
    "genre": "Fiction"
  },
  {
    "title": "The Great Gatsby",
    "author": "F. Scott Fitzgerald",
    "published_year": 1925,
    "genre": "Fiction"
  },
  {
    "title": "Pride and Prejudice",
    "author": "Jane Austen",
    "published_year": 1813,
    "genre": "Romance"
  }
]

@app.route('/')
def home():
    return "Welcome to the Book API!"

@app.route('/books', methods=['GET'])
def get_books():
    return jsonify(books)

if __name__ == '__main__':
    app.run(debug=True)
```

This script creates a simple web service with a single endpoint `/books` that returns the list of books in JSON format when accessed with an HTTP GET request.
````
