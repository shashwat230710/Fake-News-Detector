import pandas as pd

# Load the datasets
true = pd.read_csv("data/True.csv")
fake = pd.read_csv("data/Fake.csv")

# Add labels
true["label"] = "REAL"
fake["label"] = "FAKE"

# Combine datasets
df = pd.concat([true, fake], axis=0).reset_index(drop=True)

# Keep only relevant columns
df = df[["title", "text", "label"]]

# Save combined dataset
df.to_csv("data/fake_news.csv", index=False)

print("✅ New dataset saved to data/fake_news.csv with shape:", df.shape)
