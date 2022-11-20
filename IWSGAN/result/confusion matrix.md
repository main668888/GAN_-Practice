# Based on Figure_2, we decided to set the threshold at 0.55.
# The confusion matrix results are as follows.

TP, FN, TN, FP: 560, 16, 169, 32

Accuracy: (TP + TN) / (TP + TN + FP + FN) = 729 / 777 = 0.938

Sensitivity: TP / (TP + FN) = 560 / 576 = 0.972

Specificity: TN / (FP + TN) = 169 / 201 = 0.841

Precision: TP / (TP + FP) = 560 / 592 = 0.946

F1-measure: (1 + 1^2) * ((Precision * Recall) / ((1^2 * Precision) + Recall)) 
		= 1.839 / 1.918 = 0.959
