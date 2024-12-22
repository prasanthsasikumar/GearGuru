from transformers import LlamaTokenizer, LlamaForCausalLM
import torch

# Path to your downloaded model
model_path = "/Users/prasanthsasikumar/.llama/checkpoints/Llama3.2-1B-Instruct"

# Load the tokenizer and model from the local directory
tokenizer = LlamaTokenizer.from_pretrained(model_path)
model = LlamaForCausalLM.from_pretrained(model_path)

def generate_answer(question, relevant_chunks):
    """
    Generates an answer to the question based on the relevant chunks.
    """
    prompt = f"Answer the following question based on the text: {question}\n\n" + "\n\n".join(relevant_chunks)
    inputs = tokenizer(prompt, return_tensors='pt')

    # Ensure the model is in evaluation mode
    model.eval()

    with torch.no_grad():
        outputs = model.generate(inputs['input_ids'], max_length=500)
    
    # Decode the generated text to human-readable answer
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# Example usage
question = "What is the function of a carburetor in a motorcycle?"
relevant_chunks = [
    "A carburetor is an essential part of a motorcycle's engine system.",
    "It mixes air with fuel in the proper ratio for combustion."
]
print(generate_answer(question, relevant_chunks))