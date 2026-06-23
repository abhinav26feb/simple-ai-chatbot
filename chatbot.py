from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# FLAN-T5 Base model
model_name = "google/flan-t5-base"

print("Loading model! Please wait a few seconds...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

model.eval()

print("AI Assistant ready! (type 'exit' to quit)\n")

conversation_history = []

while True:
    user_input = input("> ")

    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    # Keep recent conversation history
    conversation_history = conversation_history[-10:]

    history_string = "\n".join(conversation_history)

    prompt = f"""
You are a helpful, knowledgeable, and friendly AI assistant.

Previous Conversation:
{history_string}

User Question: {user_input}

Answer:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False
    )

    response = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    ).strip()

    print("Bot:", response)

    conversation_history.append(f"User: {user_input}")
    conversation_history.append(f"Bot: {response}")
