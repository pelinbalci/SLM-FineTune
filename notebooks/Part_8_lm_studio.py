from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="C:/Users/pelin/.lmstudio/models/pelinbalci/my-qwen-finetuned-gguf/qwen2.5-0.5b-instruct.Q8_0.gguf",  # Check LM Studio for exact name
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is machine learning?"}
    ]
)

print(response.choices[0].message.content)

"""

Machine Learning (ML) is a field of Artificial Intelligence (AI) that allows computers to learn from and improve through experience without being explicitly programmed.

In simple terms, ML algorithms identify patterns in data which can help them make predictions or decisions based on new data. The goal is for the algorithm to "learn" how best to predict or decide given different inputs.

ML has a wide range of applications including fraud detection, medical diagnosis, recommendation systems, voice recognition, image processing and so forth.

At its core, ML uses statistical models that learn from training data in order to make predictions based on new input. The algorithms are usually trained using supervised learning techniques where a labeled dataset is used as the input for classification or regression problems, while unsupervised learning algorithms use data to discover patterns in unlabelled data.

ML systems can be highly sophisticated and complex with multiple layers of processing and analysis involved. It also has some limitations including that it cannot learn from experience and must be manually programmed and trained before it can make any predictions or decisions.

"""