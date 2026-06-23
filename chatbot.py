from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# BlenderBot conversational model

model_name = "facebook/blenderbot-400M-distill"

print("Loading model! Please wait a few seconds...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

print("Chatbot ready! (type 'exit' to quit)\n")

conversation_history = []

while True:
    user_input = input("> ")

    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    # Keep only the latest conversation turns
    conversation_history = conversation_history[-6:]

    history_string = "\n".join(conversation_history)

    prompt = history_string + f"\nUser: {user_input}\nBot:"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=60,
        no_repeat_ngram_size=3,
        repetition_penalty=1.3,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

    response = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    ).strip()

    print("Bot:", response)

    conversation_history.append(f"User: {user_input}")
    conversation_history.append(f"Bot: {response}")




