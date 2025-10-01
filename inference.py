import json
import requests

def ask_llm(prompt):
    # Ollama local API example
    r = requests.post("http://localhost:11434/api/generate",
                      json={"model":"llama3.2:1b","prompt":prompt,"stream":False})
    return r.json()["response"]


user_text = "Hey how can you help me?"
reply = ask_llm(f"User said: {user_text}\nRespond briefly and clearly.")
print(reply)
