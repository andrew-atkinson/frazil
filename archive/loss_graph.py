import pandas as pd, matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_csv("data/models/monthly/metrics.csv")
ax = df.dropna(subset=["train/loss"]).plot(x="step", y="train/loss")
df.dropna(subset=["val/loss"]).plot(x="step", y="val/loss", ax=ax)

folder_path = Path("plots")
folder_path.mkdir(parents=True, exist_ok=True)

plt.savefig("plots/loss.png")   # or plt.show()