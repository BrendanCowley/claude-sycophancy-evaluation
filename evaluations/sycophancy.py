from utils.claude import Claude_Conversation
import json
import datetime
import copy
from utils.evaluate import compare_embeddings, LLMaaJ_confidence, NLI_comparison


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

    # get LLM output for dataset
    for i, record in enumerate(sycophancy_dataset["records"]):
        print(f"generating output for record {i + 1}...")

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

            result[follow_up_key] = {
                "follow_up": follow_up,
                "response": follow_up_response,
                "response_history": conversation.response_history
            }

        results["results"].append(result)

    print("finished generating output! Starting evaluation of results")
    
    # run evaluation of results
    for i, result in enumerate(results["results"]):
        print(f"evaluating output for record {i + 1}...")

        for follow_up_key in follow_up_keys:
            question = result["question"]
            initial_response = result["initial_response"]["response_text"]
            follow_up_response = result[follow_up_key]["response"]["response_text"]
            follow_up = result[follow_up_key]["follow_up"]

            nli_forward, nli_backward = NLI_comparison(initial_response, follow_up_response)
            answer_movement, user_directed_movement = compare_embeddings(initial_response, follow_up_response, follow_up)        
            llmaj = LLMaaJ_confidence(question, initial_response, follow_up_response)

            result[follow_up_key]["evaluation"] = {
                "forward_contradiction": nli_forward,
                "backward_contradiction": nli_backward,
                "answer_movement": answer_movement.item(),
                "user_directed_movement": user_directed_movement.item(),
                "llmaj": {
                    "initial": llmaj[0]["confidence_score"],
                    "follow_up": llmaj[1]["confidence_score"],
                    "change": llmaj[2]["confidence_change_score"],
                    "position_changed": llmaj[2]["position_changed"]
                }
            }

    return results


if __name__ == "__main__":
    # model_id = "claude-haiku-4-5"
    model_id = "claude-opus-4-7"
    dataset_location = "./data/sycophancy_dataset.json"
    results_location = "./results/sycophancy_results.json"

    results = run_experiment(dataset_location, model_id)

    with open(results_location, "w") as f:
        json.dump(results, f)

    print("finished!")
