# Using GGUF Models Locally

## Method 1: Direct Pull from HuggingFace (Easiest)

Ollama can pull GGUF models directly from HuggingFace.

If your GGUF is already on HuggingFace, Ollama can pull it directly:


    C:\Users\pelin>ollama run hf.co/pelinbalci/my-qwen-finetuned-gguf:Q8_0
    pulling manifest
    pulling da1f54f6647d: 100% ▕██████████████████████████████████████████████████████████▏ 531 MB
    pulling e94a8ecb9327: 100% ▕██████████████████████████████████████████████████████████▏ 1.6 KB
    pulling d7da269a6d1b: 100% ▕██████████████████████████████████████████████████████████▏  475 B
    verifying sha256 digest
    writing manifest

    Use Ctrl + d or /bye to exit.
    >>> what is machine learning?
    Machine Learning (ML) is a field of computer science that focuses on the development of algorithms and statistical
    models that allow computers to improve their performance or make decisions based on data. It involves training
    machines on large amounts of data, then allowing them to learn patterns and relationships within that data using
    sophisticated algorithms.

    Some key characteristics of Machine Learning include:
    
    1. Data-driven: ML is primarily designed to work with data from the real world, enabling it to analyze and draw
    conclusions about patterns in large datasets.
       2. Self-learning: Unlike traditional programming languages like Python or Java, ML requires humans to manually
       define rules and structures for the algorithms used.
       3. Learning without explicit instruction: ML models can be trained using a combination of raw data, annotated
       examples, and labeled data to improve their performance on new tasks.
    
    Machine Learning is widely applied in areas such as:
    
    1. Image recognition and computer vision
       2. Natural language processing (NLP)
       3. Predictive analytics
       4. Fraud detection
       5. Recommendation systems
    
    One popular application of machine learning is natural language processing, where models can help machines
    understand and respond to human language by analyzing text data.
    
    In summary, Machine Learning is a field that uses algorithms and statistical models to train computers to improve
    their performance based on large amounts of raw data. It involves both self-learning and explicit instruction from
    humans, allowing it to work well in real-world applications.
    
    >>> Send a message (/? for help)


**What Ollama Does Behind the Scenes:**

- Step 1: Locate the GGUF File

Ollama looks at the HuggingFace repo and finds:

    qwen2.5-0.5b-instruct.Q8_0.gguf (~530 MB for Q8_0 of 0.5B model)

- Step 2: Download Location

Ollama downloads to its local cache:

    OSLocationmacOS~/.ollama/models/Linux~/.ollama/models/WindowsC:\Users\<user>\.ollama\models\

- Step 3: Auto-Generate Modelfile

Since Unsloth included an Ollama.modelfile in the repo, Ollama uses that. Otherwise, it auto-detects the architecture (Qwen2) and applies the correct chat template.

- Step 4: Register the Model

The model gets registered as hf.co/pelinbalci/my-qwen-finetuned-gguf


    C:\Users\pelin>ollama list
    NAME                                            ID              SIZE      MODIFIED
    hf.co/pelinbalci/my-qwen-finetuned-gguf:Q8_0    3554dc0ff480    531 MB    21 minutes ago
    mxbai-embed-large:latest                        468836162de7    669 MB    2 weeks ago
    llama3.2:latest                                 a80c4f17acd5    2.0 GB    2 weeks ago

## Method 2: LM Studio  


**Step 1: Load Your Model**

Download from HuggingFace (Inside LM Studio)

![image](images/lmstudio_1.PNG)

- Click "Select a model to load" at the top (or press Ctrl+L)
- In the search bar, type: pelinbalci/my-qwen-finetuned-gguf

![image](images/lmstudio_2.PNG)

- Select the model and click Download

![image](images/lmstudio_3.PNG)

- Once downloaded, click Load
- 
![image](images/lmstudio_4.PNG)

**Step 2: Start Chatting**

Once loaded:

Click the chat icon (top of left sidebar - the speech bubble)

![image](images/lmstudio_5.PNG)

Type your message and chat with your model!

**Step 3: Use the API**


![image](images/lmstudio_6.PNG)

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
