from transformers import AutoTokenizer, AutoModel
import numpy as np
import json
import plotly.figure_factory as ff

# Load profile interests data
with open("interests.json", "r") as file:
    profiles_data = json.load(file)

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")


# Function to generate embeddings
def generate_embeddings(text):
    inputs = tokenizer(
        text, return_tensors="pt", padding=True, truncation=True, max_length=512
    )
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings


# Generate embeddings for each profile's "looking_for" and "can_give"
embeddings = {}
for name, info in profiles_data.items():
    embeddings[name] = {
        "looking_for": generate_embeddings(info["looking_for"]),
        "can_give": generate_embeddings(info["can_give"]),
    }


# Calculate cosine similarity between each "looking_for" and "can_give"
def cosine_similarity(a, b):
    return (a @ b.T) / (a.norm() * b.norm())


# Store similarity scores
scores = {name: {} for name in profiles_data}

# Calculate similarity scores between each pair of profiles
for name1, data1 in embeddings.items():
    for name2, data2 in embeddings.items():
        if name1 != name2:
            score = cosine_similarity(data1["looking_for"], data2["can_give"]).item()
            scores[name1][name2] = score

# Print best matches based on scores
for name, matches in scores.items():
    best_match = max(matches, key=matches.get)
    print(
        f"Best match for {name} is {best_match} with a score of {matches[best_match]:.2f}"
    )

# Create a confusion matrix of the scores
names = list(profiles_data.keys())
confusion_matrix = np.zeros((len(names), len(names)))

for i, name1 in enumerate(names):
    for j, name2 in enumerate(names):
        confusion_matrix[i, j] = round(scores[name1].get(name2, 0), 2)

# Generate a heatmap for the confusion matrix
fig = ff.create_annotated_heatmap(
    confusion_matrix, x=names, y=names, colorscale="Viridis"
)
fig.update_layout(
    title="Profile Match Scores",
    xaxis=dict(title="Can Give"),
    yaxis=dict(title="Looking For"),
)
fig.write_image("match_heatmap.png")
fig.show()
