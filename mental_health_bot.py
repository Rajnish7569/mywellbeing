from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the model and tokenizer
model_name = "thrishala/mental_health_chatbot"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def get_mental_health_reply(user_input: str) -> str:
    # Tokenize input
    inputs = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")

    # Generate response
    reply_ids = model.generate(inputs, max_length=100, pad_token_id=tokenizer.eos_token_id)
    reply = tokenizer.decode(reply_ids[:, inputs.shape[-1]:][0], skip_special_tokens=True)

    return reply
