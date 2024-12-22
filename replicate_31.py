import replicate

# The meta/meta-llama-3.1-405b-instruct model can stream output as it's running.
for event in replicate.stream(
    "meta/meta-llama-3.1-405b-instruct",
    input={
        "top_k": 50,
        "top_p": 0.9,
        "prompt": "How long will be the trip?",
        "max_tokens": 1024,
        "min_tokens": 0,
        "temperature": 0.6,
        "system_prompt": "You are a helpful assistant.",
        "prompt_template": "you are an assistant that provides answers to questions based on a given context. If you can't answer the question, reply \"I don't know\". Be as concise as possible and go straight to the point. Context : \"I'm Prasanth and I will be traveling in Indian for the next four months on a motorcycle.\"\nQuestion: {prompt}",
        "presence_penalty": 0,
        "frequency_penalty": 0
    },
):
    print(str(event), end="")