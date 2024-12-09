from transformers import T5Tokenizer, T5ForConditionalGeneration

model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def generate_answer(question, context):
    input_text = f"question: {question} context: {context}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    output = model.generate(input_ids)
    return tokenizer.decode(output[0], skip_special_tokens=True)
