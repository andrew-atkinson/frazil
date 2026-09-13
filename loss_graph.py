import pandas as pd, matplotlib.pyplot as plt
df = pd.read_csv("data/models/monthly/metrics.csv")
ax = df.dropna(subset=["train/loss"]).plot(x="step", y="train/loss")
df.dropna(subset=["val/loss"]).plot(x="step", y="val/loss", ax=ax)
plt.savefig("loss.png")   # or plt.show()