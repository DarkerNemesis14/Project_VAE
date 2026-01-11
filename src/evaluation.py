import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.manifold import TSNE

class Evaluation:
    def tSNE(self, x, clusters, target_file):
        # t-SNE embedding
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        x_tsne = tsne.fit_transform(x)

        # Plot
        plt.figure(figsize=(8,6))
        scatter = plt.scatter(x_tsne[:,0], x_tsne[:,1], c=clusters, cmap='tab10', s=15)
        plt.legend(*scatter.legend_elements(), title="Clusters")
        plt.title("t-SNE Visualization of Clusters")
        plt.savefig(target_file, format="pdf", bbox_inches="tight")
        # plt.show()
        plt.close()

    def silhouette(self, features, clusters):
        return silhouette_score(features, clusters)

    def calinski_harabasz(self, features, clusters):
        return calinski_harabasz_score(features, clusters)
