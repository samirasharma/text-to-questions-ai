from transformers import pipeline

# Load the model once at module level to avoid reloading it every time
generator = pipeline(
    "text2text-generation",
    model="iarfmoose/t5-base-question-generator",
    device=-1
)

def generate_questions(text: str) -> list[str]:
    """
    Generate questions from a given passage using a pretrained T5 model.
    """
    prompt = f"generate questions: {text}"
    results = generator(
        prompt,
        max_length=64,
        num_beams=4,
        num_return_sequences=3,
        early_stopping=True
    )
    questions = [result["generated_text"] for result in results]
    return questions
