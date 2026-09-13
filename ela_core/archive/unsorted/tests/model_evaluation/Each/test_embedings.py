import numpy as np
emb_rgb = np.load("D:/Diplom/ela_core/models/All/test_embeddings_rgb_512d.npy")
emb_ela = np.load("D:/Diplom/ela_core/models/All/test_embeddings_ela_512d.npy")

print("RGB norm mean/std:", np.linalg.norm(emb_rgb, axis=1).mean(), np.linalg.norm(emb_rgb, axis=1).std())
print("ELA norm mean/std:", np.linalg.norm(emb_ela, axis=1).mean(), np.linalg.norm(emb_ela, axis=1).std())