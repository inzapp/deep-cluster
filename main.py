from deep_cluster import DeepCluster

if __name__ == '__main__':
    DeepCluster(
        train_image_path=r'.',
        cluster_output_save_path=r'./clustered',
        input_shape=(28, 28, 1),
        encoding_dim=8,
        lr=1e-3,
        epochs=20,
        batch_size=32,
        num_cluster_classes=10,
        cluster_epsilon=1e-9).cluster(use_saved_model=False)
