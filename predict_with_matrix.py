import os
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from models import TransferLearningResNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the class names (should match the classes from training)
class_names = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

# Define image transformations (should match the ones used during training)
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Ensure this matches your model's input size
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# Function to load the model
def load_model(model_path):
    model = TransferLearningResNet(num_classes=len(class_names))  # Adjust architecture as needed
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.to(device)
    model.eval()  # Set model to evaluation mode
    return model


# Function to predict the class of a single image with confidence scores
def predict_image_with_confidence(model, image_path):
    image = Image.open(image_path).convert('RGB')  # Load the image and convert to RGB
    image = image_transforms(image)  # Apply the same transforms as training
    image = image.unsqueeze(0)  # Add batch dimension

    image = image.to(device)  # Move image to device
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        predicted_class = class_names[predicted.item()]
        confidence_score = confidence.item()

    return predicted_class, confidence_score, probabilities.cpu().numpy()[0]


# Function to extract true label from filename
def get_true_label(filename):
    if 'gl' in filename.lower():
        return 'glioma_tumor'
    elif 'me' in filename.lower():
        return 'meningioma_tumor'
    elif 'no' in filename.lower():
        return 'no_tumor'
    elif 'pi' in filename.lower():
        return 'pituitary_tumor'
    else:
        return 'unknown'


# Function to create prediction matrix and visualizations
def create_prediction_matrix(model, folder_path):
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    # Limit to first 50 images for better visualization
    image_files = image_files[:50]
    
    predictions = []
    true_labels = []
    confidence_scores = []
    all_probabilities = []
    
    print(f"Processing {len(image_files)} images...")
    
    for i, image_file in enumerate(image_files):
        image_path = os.path.join(folder_path, image_file)
        true_label = get_true_label(image_file)
        predicted_class, confidence, probabilities = predict_image_with_confidence(model, image_path)
        
        predictions.append(predicted_class)
        true_labels.append(true_label)
        confidence_scores.append(confidence)
        all_probabilities.append(probabilities)
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(image_files)} images")
    
    # Create prediction results DataFrame
    results_df = pd.DataFrame({
        'Image': image_files,
        'True_Label': true_labels,
        'Predicted_Label': predictions,
        'Confidence': confidence_scores,
        'Correct': [true == pred for true, pred in zip(true_labels, predictions)]
    })
    
    # Print summary statistics
    accuracy = results_df['Correct'].mean()
    print(f"\nOverall Accuracy: {accuracy:.3f}")
    print(f"Total Images: {len(image_files)}")
    print(f"Correct Predictions: {results_df['Correct'].sum()}")
    
    # Create confusion matrix
    cm = confusion_matrix(true_labels, predictions, labels=class_names)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Plot accuracy by class
    plt.subplot(2, 2, 2)
    class_accuracy = []
    for i, class_name in enumerate(class_names):
        class_mask = np.array(true_labels) == class_name
        if class_mask.sum() > 0:
            class_acc = np.array(predictions)[class_mask] == class_name
            class_accuracy.append(class_acc.mean())
        else:
            class_accuracy.append(0)
    
    bars = plt.bar(range(len(class_names)), class_accuracy)
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.title('Accuracy by Class')
    plt.xticks(range(len(class_names)), [name.replace('_tumor', '') for name in class_names], rotation=45)
    
    # Add accuracy values on bars
    for i, (bar, acc) in enumerate(zip(bars, class_accuracy)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.3f}', ha='center', va='bottom')
    
    # Plot confidence distribution
    plt.subplot(2, 2, 3)
    plt.hist(confidence_scores, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Confidence Scores')
    plt.axvline(np.mean(confidence_scores), color='red', linestyle='--', 
                label=f'Mean: {np.mean(confidence_scores):.3f}')
    plt.legend()
    
    # Plot prediction counts
    plt.subplot(2, 2, 4)
    pred_counts = pd.Series(predictions).value_counts()
    plt.pie(pred_counts.values, labels=[name.replace('_tumor', '') for name in pred_counts.index], 
            autopct='%1.1f%%', startangle=90)
    plt.title('Prediction Distribution')
    
    plt.tight_layout()
    plt.savefig('prediction_matrix_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print detailed classification report
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions, target_names=class_names))
    
    # Save detailed results to CSV
    results_df.to_csv('prediction_results.csv', index=False)
    print(f"\nDetailed results saved to 'prediction_results.csv'")
    print(f"Visualization saved to 'prediction_matrix_analysis.png'")
    
    return results_df, cm


# Main function to load model and create prediction matrix
def main():
    # Load the trained model
    model_path = "resnet_model.pth"  # Path to your trained model
    model = load_model(model_path)

    # Path to the folder with new images
    new_images_folder = "data/New"

    # Create prediction matrix and visualizations
    results_df, confusion_mat = create_prediction_matrix(model, new_images_folder)
    
    # Display first few results
    print("\nFirst 10 predictions:")
    print(results_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
