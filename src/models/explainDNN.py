import shap
import pandas as pd
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt


def explain_shap(model, features, feature_names=None, explainer_path='explainer.pkl', shap_values_path='shap_values.pkl', force_plot_index=0, summary_max_display=10):
    # Convert the input features to a PyTorch tensor with the correct data type
    features = torch.tensor(features, dtype=torch.float64)

    if torch.cuda.is_available():
        features = features.cuda()

    # Create a DeepExplainer for the PyTorch model
    explainer = shap.DeepExplainer(model, features)

    # Calculate SHAP values for all instances
    shap_values = explainer.shap_values(features)

    # Save the explainer and SHAP values using pickle
    with open(explainer_path, 'wb') as explainer_file:
        pickle.dump(explainer, explainer_file)

    with open(shap_values_path, 'wb') as shap_values_file:
        pickle.dump(shap_values, shap_values_file)

    # init the JS visualization code
    shap.initjs()

    # Assuming features is a NumPy array
    features_np = features.cpu().detach().numpy()
    """
    # Create a force plot for the specified instance
    if force_plot_index < len(features_np):
        shap.force_plot(explainer.expected_value[0], shap_values[0][force_plot_index], features_np[force_plot_index], feature_names=feature_names)
    """
    # Create a summary plot
    plt.figure(dpi=300)
    shap.summary_plot(shap_values, features_np, max_display=summary_max_display)
