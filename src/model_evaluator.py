# src/model_evaluator.py
# Refactored Code

import logging
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, log_loss
from sklearn.base import ClassifierMixin # Use ClassifierMixin
from typing import Dict, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class ModelEvaluationStrategy(ABC):
    """ Abstract base class for model evaluation strategies. """
    @abstractmethod
    def evaluate(self, model: ClassifierMixin, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """ Abstract method to evaluate a model. """
        pass

class ClassificationModelEvaluationStrategy(ModelEvaluationStrategy):
    """ Concrete strategy for Classification Model Evaluation. """
    def evaluate(self, model: ClassifierMixin, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Evaluates a classification model using Accuracy, Precision, Recall, F1-score, LogLoss, and ROC AUC.

        Args:
            model: Trained classification model instance.
            X_test: Testing data features.
            y_test: Testing data labels/target.
        Returns:
            A dictionary containing evaluation metrics.
        """
        logging.info("Evaluating classification model.")
        try:
            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            logging.info(f"Accuracy: {accuracy:.4f}")

            # Calculate precision, recall, f1-score for each class
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None, labels=model.classes_)
            metrics = {f'precision_{cls}': p for cls, p in zip(model.classes_, precision)}
            metrics.update({f'recall_{cls}': r for cls, r in zip(model.classes_, recall)})
            metrics.update({f'f1_{cls}': f for cls, f in zip(model.classes_, f1)})

            # Calculate weighted averages
            precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
            metrics['precision_weighted'] = precision_w
            metrics['recall_weighted'] = recall_w
            metrics['f1_weighted'] = f1_w

            logging.info(f"Precision (Weighted): {precision_w:.4f}")
            logging.info(f"Recall (Weighted): {recall_w:.4f}")
            logging.info(f"F1-Score (Weighted): {f1_w:.4f}")

            metrics['accuracy'] = accuracy

            # Calculate LogLoss and ROC AUC (requires probability predictions)
            try:
                y_pred_proba = model.predict_proba(X_test)
                metrics['log_loss'] = log_loss(y_test, y_pred_proba)
                logging.info(f"Log Loss: {metrics['log_loss']:.4f}")

                # ROC AUC requires multi-class handling if necessary
                if len(model.classes_) > 2:
                     # One-vs-Rest approach for multi-class AUC
                    metrics['roc_auc_ovr_weighted'] = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
                    logging.info(f"ROC AUC (OvR Weighted): {metrics['roc_auc_ovr_weighted']:.4f}")
                elif len(model.classes_) == 2:
                    # Binary classification AUC (use probability of the positive class)
                    metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba[:, 1])
                    logging.info(f"ROC AUC: {metrics['roc_auc']:.4f}")

            except AttributeError:
                logging.warning("Model does not support predict_proba. LogLoss and ROC AUC cannot be calculated.")
            except Exception as proba_e:
                 logging.warning(f"Could not calculate LogLoss/ROC AUC: {proba_e}")


            return metrics

        except Exception as e:
            logging.error(f"Error during model evaluation: {e}")
            raise

class ModelEvaluator:
    """ Context class using a ModelEvaluationStrategy. """
    def __init__(self, strategy: ModelEvaluationStrategy):
        self._strategy = strategy

    def evaluate_model(self, model: ClassifierMixin, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        logging.info(f"Executing model evaluation strategy: {self._strategy.__class__.__name__}")
        return self._strategy.evaluate(model, X_test, y_test)