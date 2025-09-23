from transformers import pipeline

nlp = pipeline("text-generation", model="gpt2-medium")
result = nlp(
    "Describe how automation and robots are affecting people's work-life balance:",
    max_length=150,
    num_return_sequences=1,
    do_sample=True,
    top_p=0.95,
    temperature=1.0,
    repetition_penalty=1.2
)
print(result[0]['generated_text'])