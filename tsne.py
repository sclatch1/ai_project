# ---------------------------
# 6) Run t-SNE
# ---------------------------
print("Running t-SNE on", feats.shape[0], "samples…")
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, init="pca")
feats_2d = tsne.fit_transform(feats)

# ---------------------------
# 7) Plot & save
# ---------------------------
plt.figure(figsize=(10,10))
scatter = plt.scatter(feats_2d[:,0], feats_2d[:,1],
                      c=labels, cmap="tab15", s=5, alpha=0.7)
plt.legend(*scatter.legend_elements(), title="Classes",
           loc="best", fontsize="small")
plt.title("t-SNE of Test Embeddings")
plt.tight_layout()
plt.savefig(args.out_file, dpi=300)
print(f"Saved t-SNE plot to {args.out_file}")