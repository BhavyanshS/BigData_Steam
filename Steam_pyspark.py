import time
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("GameAnalysis").getOrCreate()

start = time.time()
games_rdd = spark.read.csv("Dataset/games.csv", header=True, inferSchema=True).rdd
users_rdd = spark.read.csv("Dataset/users.csv", header=True, inferSchema=True).rdd
recs_rdd = spark.read.csv("Dataset/recommendations.csv", header=True, inferSchema=True).rdd
end = time.time()
print(f"[Load CSVs] Execution time: {end - start:.2f} seconds")

start = time.time()
recs_with_users = recs_rdd.map(lambda x: (x['user_id'], x))
recs_with_games = recs_rdd.map(lambda x: (x['app_id'], x))

users_dict = users_rdd.map(lambda x: (x['user_id'], x)).collectAsMap()
games_dict = games_rdd.map(lambda x: (x['app_id'], x)).collectAsMap()

recs_with_user_game = recs_rdd.map(lambda x: (x['user_id'], (x, users_dict.get(x['user_id']), games_dict.get(x['app_id']))))

end = time.time()
print(f"[Merge Datasets] Execution time: {end - start:.2f} seconds")

start = time.time()
user_review_rate = recs_with_user_game.map(lambda x: (x[0], 
(1 if x[1][0]['is_recommended'] == 'True' else 0, 1)  
)).reduceByKey(lambda a, b: (
    a[0] + b[0], a[1] + b[1]  
)).mapValues(lambda x: x[0] / x[1])  

end = time.time()
print(f"[User Behavior Correlation] Execution time: {end - start:.2f} seconds")

start = time.time()
recs_with_user_game_filtered = recs_with_user_game.filter(lambda x: x[1][2] is not None and x[1][2]['price_final'] > 0)

price_bins_rdd = recs_with_user_game_filtered.map(lambda x: (
    x[1][2]['price_final'], 
    "(0, 10]" if x[1][2]['price_final'] <= 10 else
    "(10, 30]" if x[1][2]['price_final'] <= 30 else
    "(30, 60]" if x[1][2]['price_final'] <= 60 else
    "(60, 100]"
))

price_recommendation_rdd = price_bins_rdd.map(lambda x: (x[1], 1 if x[0] else 0))
recommend_rate_by_price = price_recommendation_rdd.aggregateByKey(
    (0, 0), 
    lambda acc, value: (acc[0] + value, acc[1] + 1), 
    lambda acc1, acc2: (acc1[0] + acc2[0], acc1[1] + acc2[1])
).mapValues(lambda x: x[0] / x[1])  

end = time.time()
print(f"[Recommendation vs Price] Execution time: {end - start:.2f} seconds")

recommend_rate_by_price_list = recommend_rate_by_price.collect()

print("\nRecommendation Rate by Price Range:")
for row in recommend_rate_by_price_list:
    print(f"{row[0]}: {row[1]:.6f}")

spark.stop()