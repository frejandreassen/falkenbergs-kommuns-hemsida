import pandas as pd

# Load the large CSV file
df = pd.read_csv('output_with_embeddings.csv')

# Split the DataFrame into two smaller DataFrames
split_index = len(df) // 2
df1 = df.iloc[:split_index]
df2 = df.iloc[split_index:]

# Save the smaller DataFrames to new CSV files
df1.to_csv('output_with_embeddings_part1.csv', index=False)
df2.to_csv('output_with_embeddings_part2.csv', index=False)
