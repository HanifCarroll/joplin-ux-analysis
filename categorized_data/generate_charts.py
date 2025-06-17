import json
import matplotlib.pyplot as plt
import numpy as np


def create_pain_point_distribution_chart(data):
    """
    Generates and saves a bar chart showing the distribution of pain points by category.
    """
    # Extract the top categories and their counts
    categories = data["summary_stats"]["pain_points"]["top_categories"]

    # Sort categories for better visualization
    sorted_categories = sorted(
        categories.items(), key=lambda item: item[1], reverse=True
    )

    labels = [item[0] for item in sorted_categories]
    values = [item[1] for item in sorted_categories]

    plt.figure(figsize=(10, 8))
    bars = plt.barh(labels, values, color="skyblue")
    plt.xlabel("Number of Pain Points")
    plt.title("Distribution of Pain Points by Category")
    plt.gca().invert_yaxis()  # Display the highest value at the top

    # Add data labels to the bars
    for bar in bars:
        plt.text(
            bar.get_width() + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width()}",
            va="center",
        )

    plt.tight_layout()
    plt.savefig("pain_points_distribution.png")
    plt.close()
    print("Saved pain point distribution chart to pain_points_distribution.png")


def create_priority_matrix(data):
    """
    Generates and saves a 2x2 priority matrix plotting impact vs. frequency.
    """
    pain_points = data["categorized_pain_points"]

    # Calculate frequency (count) and impact (average priority) for each category
    category_data = {}
    priority_map = {"high": 3, "medium": 2, "low": 1}

    for point in pain_points:
        category = point["category"]
        priority = priority_map.get(point["priority"], 0)

        if category not in category_data:
            category_data[category] = {"count": 0, "total_priority": 0}

        category_data[category]["count"] += 1
        category_data[category]["total_priority"] += priority

    # Calculate average priority
    for category, values in category_data.items():
        if values["count"] > 0:
            values["avg_priority"] = values["total_priority"] / values["count"]
        else:
            values["avg_priority"] = 0

    labels = list(category_data.keys())
    frequencies = [category_data[cat]["count"] for cat in labels]
    impacts = [category_data[cat]["avg_priority"] for cat in labels]

    # Plotting the matrix
    plt.figure(figsize=(12, 10))
    plt.scatter(frequencies, impacts, s=100, alpha=0.7, edgecolors="w", c="blue")

    plt.title("Priority Matrix: Impact vs. Frequency of Pain Points")
    plt.xlabel("Frequency (Number of Mentions)")
    plt.ylabel("Impact (Average Priority Score)")

    # Add annotations for all categories
    for i, label in enumerate(labels):
        plt.annotate(
            label,
            (frequencies[i], impacts[i]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
            weight="bold",
        )

    # Add quadrant lines
    avg_freq = np.mean(frequencies)
    avg_impact = np.mean(impacts)
    plt.axvline(x=avg_freq, color="grey", linestyle="--")
    plt.axhline(y=avg_impact, color="grey", linestyle="--")

    # Add quadrant labels
    plt.text(
        avg_freq * 1.5,
        avg_impact * 1.05,
        "High Impact / High Frequency",
        fontsize=12,
        ha="center",
        color="red",
    )
    plt.text(
        avg_freq * 0.5,
        avg_impact * 1.05,
        "High Impact / Low Frequency",
        fontsize=12,
        ha="center",
        color="orange",
    )
    plt.text(
        avg_freq * 0.5,
        avg_impact * 0.95,
        "Low Impact / Low Frequency",
        fontsize=12,
        ha="center",
        color="green",
    )
    plt.text(
        avg_freq * 1.5,
        avg_impact * 0.95,
        "Low Impact / High Frequency",
        fontsize=12,
        ha="center",
        color="blue",
    )

    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig("priority_matrix.png")
    plt.close()
    print("Saved priority matrix chart to priority_matrix.png")


if __name__ == "__main__":
    try:
        with open("categorized_results.json", "r") as f:
            analysis_data = json.load(f)

        create_pain_point_distribution_chart(analysis_data)
        create_priority_matrix(analysis_data)

    except FileNotFoundError:
        print(
            "Error: categorized_results.json not found. Make sure the file is in the same directory as the script."
        )
    except json.JSONDecodeError:
        print(
            "Error: Could not decode JSON from categorized_results.json. Please check the file format."
        )
