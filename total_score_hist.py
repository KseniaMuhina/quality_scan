import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('result.csv')

plt.hist(df['total_score'], bins=20)
plt.title('Histogram of Total Scores')
plt.xlabel('Total Score')
plt.ylabel('Frequency')
plt.savefig('histogram_total_score.png')
print("Hist saved in 'histogram_total_score.png'.")
