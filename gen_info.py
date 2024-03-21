from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline
import torch
import json

# Load the profile data
from data import profiles

# Load the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
model = GPT2LMHeadModel.from_pretrained("gpt2-medium")

# Make sure to move the model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Initialize the text generation pipeline
text_generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if device == "cuda" else -1,
)


def generate_text(prompt):
    generated_texts = text_generator(
        prompt,
        max_new_tokens=500,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
    )
    return generated_texts[0]["generated_text"]


# Function to generate "looking_for" and "can_give" texts for each profile
def generate_profile_texts(name, profile):
    if "interests" in profile:
        interests_str = ", ".join(profile["interests"])
    else:
        interests_str = profile["industry"]

    purpose_str = ", ".join(profile["purpose"])

    if "occupation" in profile:
        occupation_field_str = f"{profile['occupation']} in {profile['field']}"
    else:
        occupation_field_str = f"{profile['type']} in {profile['industry']}"

    looking_for_prompt = f"Given their interest in {interests_str} and goals of {purpose_str}, what is {name}, a {occupation_field_str}, looking for?"
    can_give_prompt = f"Considering their expertise in {interests_str}, what can {name} offer to others?"

    looking_for_text = generate_text(looking_for_prompt)
    can_give_text = generate_text(can_give_prompt)

    return looking_for_text, can_give_text


# Generate and update profiles with "looking_for" and "can_give" texts
generated = {}
for name, profile in profiles.items():
    looking_for, can_give = generate_profile_texts(name, profile)
    generated[name] = {}

    generated[name]["looking_for"] = looking_for
    generated[name]["can_give"] = can_give

# Save the updated profiles
with open("generated_interests.json", "w") as f:
    json.dump(generated, f, indent=4)
