if __name__ == '__main__':
    # Import necessary libraries
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # List files under input directory (to ensure the dataset is available)
    for dirname, _, filenames in os.walk('./input/'):
        for filename in filenames:
            print(os.path.join(dirname, filename))

    # Import YOLO from the ultralytics library
    from ultralytics import YOLO

    # Train YOLOv8n model on your dataset
    model = YOLO('yolov8n.pt')  # Load a pretrained YOLOv8n model

    # Set the device to CPU
    model.to('cpu')

    # Train the model
    results = model.train(
        data='data.yaml',  # Dataset YAML file
        epochs=20,  # Number of training epochs
        imgsz=640,   # Image size
        batch=20,    # Batch size
        name='yolov8n_underwater_plastics'  # Custom name for the run
    )

    # Plot training results using Matplotlib
    def plot_training_results(results):
        # Results is a dictionary with training metrics (accuracy, loss, etc.)
        # Accessing the results
        epochs = range(len(results['metrics']['train']['loss']))

        plt.figure(figsize=(12, 8))

        # Plotting loss
        plt.subplot(2, 2, 1)
        plt.plot(epochs, results['metrics']['train']['loss'], label='Train Loss')
        plt.plot(epochs, results['metrics']['val']['loss'], label='Validation Loss')
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # Plotting Precision
        plt.subplot(2, 2, 2)
        plt.plot(epochs, results['metrics']['train']['precision'], label='Train Precision')
        plt.plot(epochs, results['metrics']['val']['precision'], label='Validation Precision')
        plt.title('Precision')
        plt.xlabel('Epochs')
        plt.ylabel('Precision')
        plt.legend()

        # Plotting Recall
        plt.subplot(2, 2, 3)
        plt.plot(epochs, results['metrics']['train']['recall'], label='Train Recall')
        plt.plot(epochs, results['metrics']['val']['recall'], label='Validation Recall')
        plt.title('Recall')
        plt.xlabel('Epochs')
        plt.ylabel('Recall')
        plt.legend()

        # Plotting mAP (mean Average Precision)
        plt.subplot(2, 2, 4)
        plt.plot(epochs, results['metrics']['train']['map'], label='Train mAP')
        plt.plot(epochs, results['metrics']['val']['map'], label='Validation mAP')
        plt.title('mAP')
        plt.xlabel('Epochs')
        plt.ylabel('mAP')
        plt.legend()

        plt.tight_layout()
        plt.show()

    # Call the function to plot training results
    plot_training_results(results)

    # Save the model
    model.save('./yolov8n_underwater_plastics.pt')
