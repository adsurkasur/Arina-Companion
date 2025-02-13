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
    # Prepare prompt with personality and memory
    context = (
        "You are Arina, a cheerful and wise AI assistant. You love helping users and chatting with them.\n\n"
    )
    context += "\n".join(memory) + "\n" if memory else ""
    context += f"User: {user_input}\nArina:"

    # Debugging logs
    print("=" * 50)
    print("📝 Context for model:")
    print(context)
    print("=" * 50)

    # Tokenize and run inference
    inputs = tokenizer(context, return_tensors="pt").to("cuda:0")

    # Debugging input tokens
    print("🔹 Tokenized Inputs:", inputs)

    with torch.no_grad():
        output_tokens = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True
        )

    # Debugging output tokens
    print("🔹 Raw Output Tokens:", output_tokens)

    # Decode model response
    response = tokenizer.decode(output_tokens[0], skip_special_tokens=True).strip()

    # Debugging raw response
    print("📝 Raw response from model:", response)

    # Remove "Arina:" prefix and unwanted text
    if "Arina:" in response:
        response = response.split("Arina:")[1].strip()

    # Remove unrelated user messages (if any)
    if "User:" in response:
        response = response.split("User:")[0].strip()

    # Enforce word limit (50 words max)
    word_limit = 50
    response_words = response.split()
    if len(response_words) > word_limit:
        response = ' '.join(response_words[:word_limit]) + '...'

    # Debugging final cleaned response
    print("✅ Final Response:", response)
    print("=" * 50)

    # Prevent blank responses
    if not response:
        response = "I'm sorry, I didn't understand that. Can you please rephrase?"

    return response
