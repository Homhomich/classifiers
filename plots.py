import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def draw_features(dataset_path, feature1, feature2):
    df = pd.read_excel(dataset_path)
    sns.lmplot(x=feature1, y=feature2, data=df, fit_reg=False, hue='result', legend=False)
    plt.legend(loc='upper right')
    plt.show()


