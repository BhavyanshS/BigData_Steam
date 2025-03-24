import pandas as pd
import os
import time

start = time.time()
games_df = pd.read_csv("Dataset/games.csv")
users_df = pd.read_csv("Dataset/users.csv")
recs_df = pd.read_csv("Dataset/recommendations.csv")
end = time.time()
print(f"[Load CSVs] Execution time: {end - start:.2f} seconds")

start = time.time()
merged_df = recs_df.merge(users_df, on="user_id", how="left")
merged_df = merged_df.merge(games_df, on="app_id", how="left")
end = time.time()
print(f"[Merge Datasets] Execution time: {end - start:.2f} seconds")

start = time.time()
most_reviewed = recs_df['app_id'].value_counts().head(10)
recommended_games = recs_df[recs_df['is_recommended'] == True]['app_id'].value_counts().head(10)
end = time.time()
print(f"[Most Reviewed/Recommended] Execution time: {end - start:.2f} seconds")

start = time.time()
grouped = recs_df.groupby("app_id")["is_recommended"].agg(['count', 'sum'])
grouped["recommend_rate"] = grouped["sum"] / grouped["count"]
top_rec_rate = grouped.sort_values(by="recommend_rate", ascending=False).head(10)
end = time.time()
print(f"[Recommendation Rate per Game] Execution time: {end - start:.2f} seconds")

start = time.time()
merged_df["user_review_rate"] = merged_df.groupby("user_id")["is_recommended"].transform("mean")
user_summary = merged_df.groupby("user_id").agg({
    "products": "first",
    "reviews": "first",
    "user_review_rate": "mean"
})
end = time.time()
print(f"[User Behavior Correlation] Execution time: {end - start:.2f} seconds")

start = time.time()
merged_df = merged_df[merged_df["price_final"] > 0]
price_bins = pd.cut(merged_df["price_final"], bins=[0, 10, 30, 60, 100])
rec_by_price = merged_df.groupby(price_bins)["is_recommended"].mean()
end = time.time()
print(f"[Recommendation vs Price] Execution time: {end - start:.2f} seconds")

print("\nMost Reviewed Games (app_id and count):\n", most_reviewed)
print("\nMost Recommended Games (app_id and count):\n", recommended_games)
print("\nTop 10 Games by Recommendation Rate:\n", top_rec_rate)
print("\nRecommendation Rate by Price Range:\n", rec_by_price)