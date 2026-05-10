from utils.claude import Claude_Conversation
import json
import datetime
import copy


def run_experiment(dataset_location, model_id):
    with open(dataset_location, 'r') as file:
        sycophancy_dataset = json.load(file)

    results = {
        "model": model_id,
        "start time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": dataset_location,
        "results": []
    }

    follow_up_keys = [
        "light follow-up",
        "medium follow-up",
        "heavy follow-up"
    ]

    for record in sycophancy_dataset["records"]:
        initial_conversation = Claude_Conversation(model_id=model_id)
        initial_response = initial_conversation.message(record["question"])


        result = {
            "question_id": record["id"],
            "question": record["question"],
            "initial_response": initial_response,
            "type": record["type"],
            "difficulty": record["difficulty"],
            "correct answer": record["correct answer"]
        }

        initial_messages = copy.deepcopy(initial_conversation.messages)

        for follow_up_key in follow_up_keys:
            conversation = Claude_Conversation(model_id=model_id)
            conversation.messages = copy.deepcopy(initial_messages) # reset to initial model answer

            follow_up = record[follow_up_key]
            follow_up_response = conversation.message(follow_up)

            result[follow_up_key] = follow_up_response

        results["results"].append(result)

    return results


if __name__ == "__main__":
    model_id = "claude-haiku-4-5"
    dataset_location = "./data/sycophancy_dataset.json"
    results_location = "./results/sycophancy_results.json"

    results = run_experiment(dataset_location, model_id)

    with open(results_location, "w") as f:
        json.dump(results, f)
