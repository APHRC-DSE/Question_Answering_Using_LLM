import requests
url = "http://localhost:8000/receive_string/"

payload = {

  "string": "your_string"
}

r = requests.post(url, json=payload)

print(r)