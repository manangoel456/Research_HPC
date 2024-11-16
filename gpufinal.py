if __name__ == '__main__':
    # Import necessary libraries here
    # Your training code goes here
    # Import necessary libraries
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import torch  # For checking CUDA availability

    # List files under input directory (to ensure the dataset is available)
    for dirname, _, filenames in os.walk('./input/'):
        for filename in filenames:
            print(os.path.join(dirname, filename))

    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    results = None  # Initialize results as None

    try:
        # Import YOLO from the ultralytics library
        from ultralytics import YOLO

        # Load a pretrained YOLOv8n model and move it to the appropriate device
        model = YOLO('yolov8n.pt').to(device)

        # Train the model
        results = model.train(
            data='data.yaml',  # Dataset YAML file
            epochs=20,  # Number of training epochs
            imgsz=640,   # Image size
            batch=20,    # Batch size
            name='yolov8n_underwater_plastics',  # Custom name for the run
            device=device  # Ensure the model trains on the correct device (CUDA if available)
        )

    except Exception as e:
        print(f"Error during model training: {e}")

    # Function to plot training results using Matplotlib
    def plot_training_results(results):
        if results is None:
            print("No results to plot.")
            return

        try:
            # Results is a dictionary with training metrics (accuracy, loss, etc.)
            metrics = results.metrics  # Access the training metrics

            epochs = range(len(metrics['train']['loss']))

            plt.figure(figsize=(12, 8))

            # Plotting loss
            plt.subplot(2, 2, 1)
            plt.plot(epochs, metrics['train']['loss'], label='Train Loss')
            plt.plot(epochs, metrics['val']['loss'], label='Validation Loss')
            plt.title('Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()

            # Plotting Precision
            plt.subplot(2, 2, 2)
            plt.plot(epochs, metrics['train']['precision'], label='Train Precision')
            plt.plot(epochs, metrics['val']['precision'], label='Validation Precision')
            plt.title('Precision')
            plt.xlabel('Epochs')
            plt.ylabel('Precision')
            plt.legend()

            # Plotting Recall
            plt.subplot(2, 2, 3)
            plt.plot(epochs, metrics['train']['recall'], label='Train Recall')
            plt.plot(epochs, metrics['val']['recall'], label='Validation Recall')
            plt.title('Recall')
            plt.xlabel('Epochs')
            plt.ylabel('Recall')
            plt.legend()

            # Plotting mAP (mean Average Precision)
            plt.subplot(2, 2, 4)
            plt.plot(epochs, metrics['train']['map'], label='Train mAP')
            plt.plot(epochs, metrics['val']['map'], label='Validation mAP')
            plt.title('mAP')
            plt.xlabel('Epochs')
            plt.ylabel('mAP')
            plt.legend()

            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Error during plotting: {e}")

    # Call the function to plot training results if results are defined
    plot_training_results(results)

    try:
        # Save the model if training was successful
        if results is not None:
            model.save('./yolov8n_underwater_plastics.pt')

    except Exception as e:
        print(f"Error during model saving: {e}")
