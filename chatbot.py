import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

def generate_response(prompt, model, tokenizer, max_length=1024):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    attention_mask = torch.ones(inputs.shape, dtype=torch.long)

    # Limit input length
    if inputs.shape[1] > tokenizer.model_max_length:
        inputs = inputs[:, -tokenizer.model_max_length:]

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            attention_mask=attention_mask,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            early_stopping=False
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def chat(model_dir='./results'):
    # Load tokenizer and model from the specified directory
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    
    print("Chatbot is ready! Type 'exit' to stop the conversation.")
    
    chat_history = ""

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        
        chat_history += f"You: {user_input}\n"
        response = generate_response(chat_history, model, tokenizer)
        response_text = response.split("You:")[-1].strip()  # Get only the model's response
        chat_history += f"AI: {response_text}\n"
        
        print(f"AI: {response_text}")

if __name__ == '__main__':
    chat()  # Default to loading from './results'

