# src/pipeline/fine_tune/fine_tune.py

import os
import json
from typing import List, Dict

import openai


# ------------------------------------------------------------------------------
# FINE-TUNING STEP FOR YOUR LANGUAGE MODEL
# ------------------------------------------------------------------------------
# This script prepares training data, performs fine-tuning, and saves the fine-tuned model's IDs.
#
# ------------------------------------------------------------------------------
# Format (OpenAI fine-tuning) per row in fine_tune_data.jsonl:
#
# {"messages":[
#   {"role":"system","content":"You are a helpful and accurate Customer Support Bot."},
#   {"role":"user","content":"What is your return policy?"},
#   {"role":"assistant","content":"We have a 30-day return policy with a full refund."}
# ]}
# ------------------------------------------------------------------------------

def prepare_finetune_data(input_file: str, output_file: str) -> None:
    """
    Prepare training data for fine-tuning by following fine-tuning format.

    Args:
        input_file (str): Source training data (JSON).
        output_file (str): File to save processed training data in fine-tuning format.
    """
    with open(input_file, "r") as infile:
        raw_data = json.load(infile)

    training_data = []

    for item in raw_data:
        dialogue = item['dialogue']

        messages = [{"role": "system", "content": "You are a helpful and accurate Customer Support Bot."}]

        for msg in dialogue:
            messages.append({"role": msg['role'], "content": msg['content']})

        training_data.append({"messages": messages})

    with open(output_file, "w") as outfile:
        for entry in training_data:
            outfile.write(json.dumps(entry) + '\n')

    print(f"Training data successfully prepared and saved to {output_file}")


def fine_tune_model(output_file: str, base_model: str = "gpt-4",
                    n_epochs: int = 5, batch_size: int = 32) -> str:
    """
    Fine-tune the base large language model with custom data.

    Args:
        output_file (str): File path to training data in fine-tuning format.
        base_model (str): The base LLM to fine-tune upon.
        n_epochs (int): Number of training epochs.
        batch_size (int): Size of each training batch.

    Returns:
        str: The fine-tuned model identifier.
    """
    # Uncomment this section when you have configured your API key
    fine_tuned = openai.FineTune.create(
        training_file=output_file,
        model=base_model,
        n_epochs=n_epochs,
        batch_size=batch_size,
    )
    fine_tuned_model = fine_tuned['id']

    fine_tuned_model = f"{base_model}_fine_tuned_{n_epochs}_{batch_size}"

    print(f"Model fine-tuned. Model id: {fine_tuned_model}")

    return fine_tuned_model


def evaluate_model(model_id: str, test_file: str) -> float:
    """
    Evaluate fine-tuned model against a test set.

    Args:
        model_id (str): The fine-tuned model's ID.
        test_file (str): File with test data in format:
                         [{"input": "...", "expected_output": "..."}, ...]

    Prints accuracy to console.
    """
    with open(test_file, "r") as f:
        test_data = json.load(f)

    correct = 0
    total = len(test_data)

    for item in test_data:
        # Query fine-tuned model
        response = openai.ChatCompletion.create(
            model=model_id,
            messages=[
                {"role": "user", "content": item['input']}
            ],
            temperature=0
        )
        output = response.choices[0].message['content'].strip()

        if output == item['expected_output'].strip():
            correct += 1

    accuracy = correct / total
    print(f"Model accuracy: {accuracy * 100:.2f}%")

    return accuracy


def main():
    """
    Main fine-tuning pipeline.
    """
    # File with raw training conversations
    input_file = "data/fine_tune_data_raw.json"

    # File to create fine-tuning training data
    output_file = "data/fine_tune_data.jsonl"

    # File with evaluation data
    test_file = "data/fine_tune_validation_data.jsonl"

    # Prepare training data first
    prepare_finetune_data(input_file, output_file)

    # Perform fine-tuning
    fine_tuned_model = fine_tune_model(output_file)

    # Evaluate fine-tuned model
    evaluate_model(fine_tuned_model, test_file)


if __name__ == "__main__":
    main()
