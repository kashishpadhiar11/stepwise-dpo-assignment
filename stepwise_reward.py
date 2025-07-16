from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def load_evaluator_model(model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

def evaluate_stepwise_solution(solution_steps, eval_pipe):
    scores = []
    for step in solution_steps:
        prompt = f"Rate the following math solution step as 'Great', 'Okay', or 'Bad':\nStep: {step}\nRating:"
        result = eval_pipe(prompt, max_new_tokens=5)[0]['generated_text']
        score = parse_rating(result)
        scores.append(score)
    return scores

def parse_rating(result_text):
    # Extract 'Great', 'Okay', or 'Bad' and map to numeric score
    result_text = result_text.lower()
    if "great" in result_text:
        return 2
    elif "okay" in result_text:
        return 1
    elif "bad" in result_text:
        return 0
    else:
        return -1  # Unknown/Unsure
