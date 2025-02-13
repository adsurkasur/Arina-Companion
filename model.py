from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM
import torch

# Load model and tokenizer once at startup
model_name_or_path = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
model = AutoGPTQForCausalLM.from_quantized(
    model_name_or_path,
    use_safetensors=True,
    device="cuda:0",
    use_triton=False
)

def generate_response(user_input, memory):
    """Generate a response from Arina based on user input and memory."""
    context = f"You are Arina, a cheerful and wise AI assistant. You love helping users and chatting with them.\n\n"
    context += "\n".join(memory) + "\n"
    context += f"User: {user_input}\nArina:"

    # Debugging context before sending to model
    print("📝 Context for Model:\n", context)

    # Tokenize and run inference
    inputs = tokenizer(context, return_tensors="pt").to("cuda:0")
    print("🔢 Tokenized Input:", inputs)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=100, temperature=0.7, do_sample=True)
    
    print("📜 Raw Output Tokens:", output)

    response = tokenizer.decode(output[0], skip_special_tokens=True).strip()

    # Debugging raw response from the model
    print("🗨️ Raw Response from Model:", response)

    # Ensure the response does not contain unwanted text
    if response.startswith("Arina:"):
        response = response.replace("Arina:", "").strip()

    response = response.split("User:")[0].strip()

    # Enforce word limit
    word_limit = 50
    response_words = response.split()
    if len(response_words) > word_limit:
        response = ' '.join(response_words[:word_limit]) + '...'

    # Final Debugging
    print("✅ Processed Response:", response)

    return response
