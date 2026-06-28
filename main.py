from src.dataset_loader import load_sst2
from src.prompts import SENTIMENT_PROMPT
from src.groq_runner import MODELS, get_response
from src.parser import parse_sentiment
from src.evaluator import evaluate


print("Loading dataset...")

dataset = load_sst2(sample_size=100)

print(f"Loaded {len(dataset)} examples")


results = {}

for model_name in MODELS:

    print("\n" + "="*50)
    print(f"Evaluating {model_name}")
    print("="*50)

    y_true = []
    y_pred = []

    for example in dataset:

        text = example["sentence"]
        true_label = example["label"]

        prompt = SENTIMENT_PROMPT.format(
            text=text
        )

        response = get_response(
            prompt,
            model_name
        )

        predicted_label = parse_sentiment(
            response
        )

        y_true.append(true_label)
        y_pred.append(predicted_label)

    metrics = evaluate(
        y_true,
        y_pred
    )

    results[model_name] = metrics

    print("\nResults")

    for metric, value in metrics.items():
        print(
            f"{metric}: {value:.4f}"
        )

print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)

print(f"{'Model':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
print("-"*60)

for model, metrics in results.items():

    print(
        f"{model:<10}"
        f"{metrics['accuracy']:<10.4f}"
        f"{metrics['precision']:<10.4f}"
        f"{metrics['recall']:<10.4f}"
        f"{metrics['f1']:<10.4f}"
    )