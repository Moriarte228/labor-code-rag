import os
import requests
from dotenv import load_dotenv
from load_document import get_a_prompt, retrieve_relevant, chunks

load_dotenv() 

api_key = os.environ.get("OPENROUTER_API_KEY")

headers = {
    "Authorization": f"Bearer {api_key}"
}

print("Введите свой вопрос для консультанта:")
query = input()
list_of_relevant_chunks = retrieve_relevant(query)
prompt = get_a_prompt(query)
print(prompt)

def send_the_message(message):
    message_body = {
        'model': 'openai/gpt-oss-120b:free',
        "messages": [{"role": "user", "content": f"{message}"}]
    }
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=message_body
    )
    data = response.json()
    if "choices" not in data:
        return f"ERROR {data['error']['code']}: {data['error']['message']}"
    return data["choices"][0]["message"]["content"]



message = send_the_message(prompt)


print(message)