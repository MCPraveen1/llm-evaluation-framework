# src/groq_runner.py

from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)   

MODELS = {

    "llama31":
    "llama-3.1-8b-instant",

    "llama33":
    "llama-3.3-70b-versatile"
}

def get_response(prompt, model_name):

    response = client.chat.completions.create(

        model=MODELS[model_name],

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0.01
    )

    return response.choices[0].message.content