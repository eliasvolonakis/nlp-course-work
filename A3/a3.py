import pandas as pd
from pyabsa import ABSAModelCheckpoint, ABSAConfig

# Load the reviews dataset
df = pd.read_csv('reviews.csv')

# Create an ABSA model using a pre-trained BERT-based model
config = ABSAConfig(task='absa', lang='multi', mode='train')
checkpoint = ABSAModelCheckpoint(config=config, checkpoint='multilingual')
model = checkpoint.load()

# Perform ABSA on the reviews
reviews = df['text'].tolist()
results = model.extract_aspect([' '.join(review.split()) for review in reviews])

# Output the results as a table
output = []
for idx, (review, result) in enumerate(zip(reviews, results)):
    name = df.loc[idx, 'product_name']
    for aspect in result.aspect_categories:
        output.append({
            'Aspect': aspect.category,
            'Sentiment': aspect.sentiment,
            'review_ID': idx,
            'Name': name
        })

output_df = pd.DataFrame(output)
print(output_df)
