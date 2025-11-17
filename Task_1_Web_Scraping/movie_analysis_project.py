import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import matplotlib.pyplot as plt # FIX for plt is not defined
import seaborn as sns           # FIX for sns is not defined
import numpy as np              # Recommended for numerical operations


sns.set_style("whitegrid")

## 🚀 Task 1: Web Scraping IMDb Movie Data

print("--- Starting Task 1: Web Scraping ---")
URL = 'https://www.imdb.com/chart/top/'
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

try:
    response = requests.get(URL, headers=HEADERS)
    soup = BeautifulSoup(response.text, 'html.parser')
    movie_containers = soup.select('li.ipc-metadata-list-summary-item')
    movie_data = []

    for container in movie_containers:
        try:
            # Extract Title and Rank
            title_tag = container.select_one('h3.ipc-title__text')
            full_title = title_tag.text.strip()
            match = re.match(r'(\d+)\.\s+(.*)', full_title)
            rank = int(match.group(1)) if match else None
            title = match.group(2) if match else full_title
            
            # Extract Year
            metadata_items = container.select('span.cli-title-metadata-item')
            year = metadata_items[0].text.strip() if len(metadata_items) > 0 else 'N/A'
            
            # Extract Rating and Votes
            rating_tag = container.select_one('span.ipc-rating-star--rating')
            imdb_rating = None
            votes = None
            if rating_tag:
                rating_text = rating_tag.text.strip()
                rating_match = re.match(r'(\d\.\d)\s*\((.*)\)', rating_text)
                if rating_match:
                    imdb_rating = float(rating_match.group(1))
                    votes_text = rating_match.group(2)
                    # Simple vote count cleanup (M/K conversion)
                    if 'M' in votes_text:
                        votes = float(votes_text.replace('M', '').replace(',', '')) * 1_000_000
                    elif 'K' in votes_text:
                        votes = float(votes_text.replace('K', '').replace(',', '')) * 1_000
                    else:
                        votes = float(votes_text.replace(',', ''))

            movie_data.append({
                'Rank': rank, 'Title': title, 'Year': year, 'IMDb_Rating': imdb_rating, 'Votes': votes
            })

        except Exception as e:
            # print(f"Skipping container due to error: {e}")
            continue

    df = pd.DataFrame(movie_data)
    df.to_csv('imdb_top_movies.csv', index=False, encoding='utf-8')
    print("✅ Task 1 Completed. Data saved to 'imdb_top_movies.csv'.")

except requests.exceptions.RequestException as e:
    print(f"Error connecting to IMDb. Skipping scraping and attempting to load existing CSV. Error: {e}")
    try:
        df = pd.read_csv('imdb_top_movies.csv')
    except FileNotFoundError:
        print("Fatal Error: Could not scrape and no existing CSV found.")
        exit()


## 📊 Task 2: Exploratory Data Analysis (EDA)

print("\n--- Starting Task 2: EDA ---")

# Data Cleaning and Type Conversion
df['Year'] = df['Year'].astype(str) # FIX: Ensure column is string before using .str accessor
df['Year'] = df['Year'].str.replace(r'[^\d]', '', regex=True)
df['Year'] = pd.to_numeric(df['Year'], errors='coerce').astype('Int64')

# Handle Missing Values (Crucial for Top 250 data)
df.dropna(subset=['Rank', 'IMDb_Rating', 'Votes', 'Year'], inplace=True)

# Final Type Conversion
df['Rank'] = df['Rank'].astype('int')
df['Votes'] = df['Votes'].astype('int')

# Feature Engineering: Create 'Decade' column
df['Decade'] = (df['Year'] // 10) * 10
df.to_csv('imdb_movies_cleaned.csv', index=False)
print(f"DataFrame cleaned and ready. Total movies: {df.shape[0]}")
print("✅ Task 2 Completed.")


## 🎨 Task 3: Data Visualization

print("\n--- Starting Task 3: Data Visualization ---")
plt.rcParams['figure.figsize'] = (10, 6) 

# --- Visualization 1: Distribution of IMDb Ratings ---
plt.figure(figsize=(10, 6))
sns.histplot(df['IMDb_Rating'], bins=10, kde=True, color='skyblue', edgecolor='black')
plt.title('Distribution of IMDb Ratings for Top Movies', fontsize=16)
plt.xlabel('IMDb Rating', fontsize=12)
plt.ylabel('Number of Movies', fontsize=12)
plt.savefig('rating_distribution.png') 
plt.show()

# --- Visualization 2: Trend of Average Rating Across Decades ---
# RE-FIX: Ensure this crucial variable is defined before use
avg_rating_by_decade = df.groupby('Decade')['IMDb_Rating'].mean().reset_index()
avg_rating_by_decade.columns = ['Decade', 'Average_Rating']

plt.figure(figsize=(12, 7))
sns.lineplot(
    x='Decade', 
    y='Average_Rating', 
    data=avg_rating_by_decade,       
    marker='o',                      
    color='darkorange',              
    linewidth=3                      
)                                    
plt.title('Average IMDb Rating Trend by Decade (Top Movies)', fontsize=16)
plt.xlabel('Decade', fontsize=12)
plt.ylabel('Average IMDb Rating', fontsize=12)
# Add data labels for clarity (optional, but good practice)
for index, row in avg_rating_by_decade.iterrows():
    plt.text(row['Decade'], row['Average_Rating'] + 0.005, 
             f"{row['Average_Rating']:.2f}", ha='center', va='bottom', fontsize=10)
plt.savefig('rating_by_decade.png') 
plt.show()

# --- Visualization 3: Relationship between Rating and Number of Votes ---
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x='IMDb_Rating', 
    y='Votes', 
    data=df, 
    alpha=0.6, 
    hue='Decade', 
    palette='viridis', 
    s=100
)
plt.yscale('log')
plt.title('IMDb Rating vs. Total Votes (Log Scale)', fontsize=16)
plt.xlabel('IMDb Rating', fontsize=12)
plt.ylabel('Number of Votes (Log Scale)', fontsize=12)
plt.legend(title='Decade')
plt.savefig('rating_vs_votes.png') 
plt.show()

print("\n✅ Task 3 Completed.")
print("\n--- Project complete.")
