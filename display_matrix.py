import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Read the prediction results
results_df = pd.read_csv('prediction_results.csv')

# Create confusion matrix
class_names = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
cm = confusion_matrix(results_df['True_Label'], results_df['Predicted_Label'], labels=class_names)

# Display the confusion matrix as a table
print("CONFUSION MATRIX:")
print("=" * 50)
print(f"{'True/Predicted':<20}", end="")
for pred in class_names:
    print(f"{pred.replace('_tumor', ''):<12}", end="")
print()

for i, true in enumerate(class_names):
    print(f"{true.replace('_tumor', ''):<20}", end="")
    for j in range(len(class_names)):
        print(f"{cm[i,j]:<12}", end="")
    print()

print("\n" + "=" * 50)

# Display accuracy metrics
print("\nACCURACY METRICS:")
print("=" * 50)
overall_accuracy = results_df['Correct'].mean()
print(f"Overall Accuracy: {overall_accuracy:.3f} ({overall_accuracy*100:.1f}%)")
print(f"Total Images: {len(results_df)}")
print(f"Correct Predictions: {results_df['Correct'].sum()}")
print(f"Wrong Predictions: {(~results_df['Correct']).sum()}")

print("\nCLASS-WISE ACCURACY:")
print("=" * 50)
for class_name in class_names:
    class_mask = results_df['True_Label'] == class_name
    if class_mask.sum() > 0:
        class_accuracy = results_df[class_mask]['Correct'].mean()
        print(f"{class_name.replace('_tumor', ''):<20}: {class_accuracy:.3f} ({class_accuracy*100:.1f}%)")

print("\nCONFIDENCE STATISTICS:")
print("=" * 50)
print(f"Mean Confidence: {results_df['Confidence'].mean():.3f}")
print(f"Min Confidence: {results_df['Confidence'].min():.3f}")
print(f"Max Confidence: {results_df['Confidence'].max():.3f}")
print(f"Std Confidence: {results_df['Confidence'].std():.3f}")

print("\nPREDICTION DISTRIBUTION:")
print("=" * 50)
pred_counts = results_df['Predicted_Label'].value_counts()
for pred, count in pred_counts.items():
    percentage = (count / len(results_df)) * 100
    print(f"{pred.replace('_tumor', ''):<20}: {count:>3} images ({percentage:>5.1f}%)")

print("\nWRONG PREDICTIONS:")
print("=" * 50)
wrong_predictions = results_df[~results_df['Correct']]
if len(wrong_predictions) > 0:
    for _, row in wrong_predictions.iterrows():
        print(f"Image: {row['Image']}")
        print(f"  True: {row['True_Label'].replace('_tumor', '')}")
        print(f"  Predicted: {row['Predicted_Label'].replace('_tumor', '')}")
        print(f"  Confidence: {row['Confidence']:.3f}")
        print()
else:
    print("No wrong predictions!")

# Create a simple visualization
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=[name.replace('_tumor', '') for name in class_names], 
            yticklabels=[name.replace('_tumor', '') for name in class_names])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

plt.subplot(2, 2, 2)
class_accuracy = []
for class_name in class_names:
    class_mask = results_df['True_Label'] == class_name
    if class_mask.sum() > 0:
        class_acc = results_df[class_mask]['Correct'].mean()
        class_accuracy.append(class_acc)
    else:
        class_accuracy.append(0)

bars = plt.bar(range(len(class_names)), class_accuracy)
plt.xlabel('Class')
plt.ylabel('Accuracy')
plt.title('Accuracy by Class')
plt.xticks(range(len(class_names)), [name.replace('_tumor', '') for name in class_names], rotation=45)
plt.ylim(0, 1.1)

# Add accuracy values on bars
for i, (bar, acc) in enumerate(zip(bars, class_accuracy)):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
            f'{acc:.3f}', ha='center', va='bottom')

plt.subplot(2, 2, 3)
plt.hist(results_df['Confidence'], bins=20, alpha=0.7, edgecolor='black')
plt.xlabel('Confidence Score')
plt.ylabel('Frequency')
plt.title('Distribution of Confidence Scores')
plt.axvline(results_df['Confidence'].mean(), color='red', linestyle='--', 
            label=f'Mean: {results_df["Confidence"].mean():.3f}')
plt.legend()

plt.subplot(2, 2, 4)
pred_counts = results_df['Predicted_Label'].value_counts()
plt.pie(pred_counts.values, labels=[name.replace('_tumor', '') for name in pred_counts.index], 
        autopct='%1.1f%%', startangle=90)
plt.title('Prediction Distribution')

plt.tight_layout()
plt.savefig('prediction_summary.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nSummary visualization saved as 'prediction_summary.png'")
