import pandas as pd
import matplotlib.pyplot as plt
import json
import os


def load_data_as_df(results_location):
    with open(results_location, 'r') as file:
        results_json = json.load(file)

    results = results_json["results"]

    followup_types = [
        "light follow-up",
        "medium follow-up",
        "heavy follow-up"
    ]
    rows = []

    for result in results:
        for followup_type in followup_types:
            evaluation = result[followup_type]["evaluation"]

            forward_contradiction = evaluation["forward_contradiction"]
            backward_contradiction = evaluation["backward_contradiction"]
            llmaj = evaluation["llmaj"]

            row = {
                "question_id": result["question_id"],
                "question_type": result["type"],
                "question_difficulty": result["difficulty"],
                "followup_strength": followup_type,
                "forward_contradiction_label": forward_contradiction["label"],
                "forward_contradiction_score": forward_contradiction["score"],
                "backward_contradiction_label": backward_contradiction["label"],
                "backward_contradiction_score": backward_contradiction["score"],
                "answer_movement": evaluation["answer_movement"],
                "user_directed_movement": evaluation["user_directed_movement"],
                "llmaj_confidence_change": llmaj["change"],
                "llmaj_position_changed":llmaj["position_changed"]
            }

            rows.append(row)

    return pd.DataFrame(rows)


def aggregate_analysis(results_df):
    dataset_wide = {
        "position_change_rate": results_df["llmaj_position_changed"].mean(),
        "confidence_change": results_df["llmaj_confidence_change"].mean(),
        "answer_movement": results_df["answer_movement"].mean(),
        "user_directed_movement": results_df["user_directed_movement"].mean()
    }

    question_type = {
        "position_change_rate": results_df.groupby("question_type")["llmaj_position_changed"].mean(),
        "confidence_change": results_df.groupby("question_type")["llmaj_confidence_change"].mean(),
        "answer_movement": results_df.groupby("question_type")["answer_movement"].mean(),
        "user_directed_movement": results_df.groupby("question_type")["user_directed_movement"].mean()
    }

    question_difficulty = {
        "position_change_rate": results_df.groupby("question_difficulty")["llmaj_position_changed"].mean(),
        "confidence_change": results_df.groupby("question_difficulty")["llmaj_confidence_change"].mean(),
        "answer_movement": results_df.groupby("question_difficulty")["answer_movement"].mean(),
        "user_directed_movement": results_df.groupby("question_difficulty")["user_directed_movement"].mean()
    }

    follow_up_strength = {
        "position_change_rate": results_df.groupby("followup_strength")["llmaj_position_changed"].mean(),
        "confidence_change": results_df.groupby("followup_strength")["llmaj_confidence_change"].mean(),
        "answer_movement": results_df.groupby("followup_strength")["answer_movement"].mean(),
        "user_directed_movement": results_df.groupby("followup_strength")["user_directed_movement"].mean()
    }

    forward_nli_distributions = {
        "all": results_df["forward_contradiction_label"].value_counts(normalize=True),
        "question_type": results_df.groupby("question_type")["forward_contradiction_label"].value_counts(normalize=True),
        "question_difficulty": results_df.groupby("question_difficulty")["forward_contradiction_label"].value_counts(normalize=True),
        "follow_up_strength": results_df.groupby("followup_strength")["forward_contradiction_label"].value_counts(normalize=True)
    }

    backward_nli_distributions = {
        "all": results_df["backward_contradiction_label"].value_counts(normalize=True),
        "question_type": results_df.groupby("question_type")["backward_contradiction_label"].value_counts(normalize=True),
        "question_difficulty": results_df.groupby("question_difficulty")["backward_contradiction_label"].value_counts(normalize=True),
        "follow_up_strength": results_df.groupby("followup_strength")["backward_contradiction_label"].value_counts(normalize=True)
    }
    
    # Agreement Rate Between nli and llmaj
    agreement = (((results_df["forward_contradiction_label"] == "contradiction") ==  (results_df["llmaj_position_changed"] == True)) ).mean()

    # Correlation Matrix
    correlation_df = results_df[
        [
            "forward_contradiction_score",
            "backward_contradiction_score",
            "answer_movement",
            "user_directed_movement",
            "llmaj_confidence_change",
            "llmaj_position_changed"
        ]
    ]

    correlation = correlation_df.corr(numeric_only=True)

    analysis_results = {
        "dataset_wide": dataset_wide,
        "question_type": question_type,
        "question_difficulty": question_difficulty,
        "follow_up_strength": follow_up_strength,
        "forward_nli_distributions": forward_nli_distributions,
        "backward_nli_distributions": backward_nli_distributions,
        "agreement_rate": agreement,
        "correlation": correlation
    }

    return analysis_results


def save_and_show(plot_directory, name):
        path = os.path.join(plot_directory, f"{name}.png")
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close("all")


def visualise(results_df, analysis_results, plot_directory):
    # Correlation Matrix Heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(analysis_results["correlation"])
    plt.colorbar()
    plt.xticks(range(len(analysis_results["correlation"].columns)), analysis_results["correlation"].columns, rotation=45)
    plt.yticks(range(len(analysis_results["correlation"].columns)), analysis_results["correlation"].columns)
    plt.title("Metric Correlation Matrix")
    save_and_show(plot_directory, "correlation_matrix")

    # Per-Question Heatmap — Answer Movement
    heatmap_data = results_df.pivot(
        index="question_id",
        columns="followup_strength",
        values="answer_movement"
    )
    heatmap_data = heatmap_data[["light follow-up", "medium follow-up", "heavy follow-up"]]

    plt.figure(figsize=(6, 10))
    plt.imshow(heatmap_data, aspect="auto")
    plt.colorbar(label="Answer Movement")
    plt.xticks(range(len(heatmap_data.columns)), heatmap_data.columns)
    plt.yticks(range(len(heatmap_data.index)), heatmap_data.index)
    plt.title("Per-Question Answer Movement")
    save_and_show(plot_directory, "question_heatmap_answer_movement")

    # Per-Question Heatmap — Position Changed
    heatmap_data = results_df.pivot(
        index="question_id",
        columns="followup_strength",
        values="llmaj_position_changed"
    )
    heatmap_data = heatmap_data[["light follow-up", "medium follow-up", "heavy follow-up"]]

    plt.figure(figsize=(6, 10))
    plt.imshow(heatmap_data, aspect="auto")
    plt.colorbar(label="Position Changed")
    plt.xticks(range(len(heatmap_data.columns)), heatmap_data.columns)
    plt.yticks(range(len(heatmap_data.index)), heatmap_data.index)
    plt.title("Per-Question Position Change")
    save_and_show(plot_directory, "question_heatmap_position_change")

    # Bar Chart — Position Change by Question Type
    analysis_results["question_type"]["position_change_rate"].plot.bar()
    plt.ylabel("Position Change Rate")
    plt.title("Position Change by Question Type")
    save_and_show(plot_directory, "bar_question_type")


    # Bar Chart — Position Change by Difficulty
    analysis_results["question_difficulty"]["position_change_rate"].plot.bar()
    plt.ylabel("Position Change Rate")
    plt.title("Position Change by Difficulty")
    save_and_show(plot_directory, "bar_difficulty")

    # Bar Chart — Position Change by Follow-Up Strength
    analysis_results["follow_up_strength"]["position_change_rate"].plot.bar()
    plt.ylabel("Position Change Rate")
    plt.title("Position Change by Follow-Up Strength")
    save_and_show(plot_directory, "bar_followup_strength")

    # Confidence Change by Follow-Up Strength
    analysis_results["follow_up_strength"]["confidence_change"].plot.bar()
    plt.ylabel("Confidence Change")
    plt.title("Confidence Change by Follow-Up Strength")
    save_and_show(plot_directory, "bar_confidence_change")

    # Boxplot — Confidence vs Position Change
    results_df.boxplot(
        column="llmaj_confidence_change",
        by="llmaj_position_changed"
    )

    plt.title("Confidence Change vs Position Change")
    plt.suptitle("")
    plt.ylabel("Confidence Change")
    save_and_show(plot_directory, "boxplot_confidence_vs_position")

    # Forward NLI Distribution
    analysis_results["forward_nli_distributions"]["all"].plot.bar()
    plt.ylabel("Proportion")
    plt.title("Forward NLI Distribution")
    save_and_show(plot_directory, "nli_forward_distribution")

    return


def analyse_results(results_location, plot_directory):
    print("loading data as dataframe...")
    results_df = load_data_as_df(results_location)

    print("performing aggregate analysis...")
    analysis_results = aggregate_analysis(results_df)

    print("generating visualisations...")
    visualise(results_df, analysis_results, plot_directory)

    return analysis_results


if __name__ == "__main__":
    #example usage
    results_location = ""
    plot_directory = ""

    analysis_results = analyse_results(results_location, plot_directory)
