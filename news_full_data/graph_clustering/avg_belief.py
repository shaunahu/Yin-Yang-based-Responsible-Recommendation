# user_beliefs is shape [num_users, num_clusters]
avg_belief = user_beliefs.mean(axis=0)

# normalise so that the 5 numbers sum to 1
avg_belief_norm = avg_belief / avg_belief.sum()

print("Average belief across clusters (normalised):")
print(avg_belief_norm.tolist())
