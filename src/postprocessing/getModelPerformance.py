import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

def plot_model_performance(num_epochs, evaluation_losses, training_losses, learning_rate):

    epochs = list(range(1, num_epochs+1))
    # Create a figure and axes
    fig, ax1 = plt.subplots()

    # Plot training losses and MAE on the same subplot

    ax1.plot(epochs, evaluation_losses, 'g-', label='Validation Loss')

    ax1.plot(epochs, training_losses, 'b-', label='Training Loss')


    # Set y-axis label for the first subplot
    ax1.set_ylabel('Loss & MAE', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    #ax1.set_ylim(0.2, 1)

    # Create a twin Axes sharing the xaxis
    ax2 = ax1.twinx()

    # Plot learning rates on the same subplot
    ax2.plot(epochs, learning_rate, 'r-', label='Learning Rate')

    # Set y-axis label for the second subplot
    ax2.set_ylabel('Learning Rate', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # Set x-axis label and title
    plt.xlabel('Epochs')
    plt.title('Training Performance')

    # Combine legend for all plots
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')

    # Adjust layout to prevent clipping of labels
    plt.tight_layout()

    # Show the plot
    plt.show()


def evaluate_model(model, testloader, output_path):
    test_targets = []
    test_labels =[]
    test_features = []
    for features, labels in testloader:
        if torch.cuda.is_available():
            features = features.cuda()
            labels = labels.cuda()

        features = features.to(dtype=torch.double)
        labels = labels.to(dtype=torch.double)   
        
        targets = model(features) #predictions
        test_targets.append(targets)
        test_labels.append(labels)
        test_features.append(features)

    # Concatenate the tensors along the specified dimension (assuming dimension 0)
    concatenated_tensor = torch.cat(test_labels, dim=0)
    # Convert the concatenated tensor to a NumPy array
    test_labels = concatenated_tensor.cpu().numpy()

    # Concatenate the tensors along the specified dimension (assuming dimension 0)
    concatenated_tensor = torch.cat(test_features, dim=0)
    # Convert the concatenated tensor to a NumPy array
    test_features = concatenated_tensor.cpu().numpy()

    # Concatenate the tensors along the specified dimension (assuming dimension 0)
    concatenated_tensor = torch.cat(test_targets, dim=0)
    # Convert the concatenated tensor to a NumPy array
    test_targets = concatenated_tensor.detach().cpu().numpy().flatten()
    
    corr_coef = np.ma.corrcoef(test_labels, test_targets)[0, 1]
    mae = mean_absolute_error(test_labels, test_targets)
    mse = mean_squared_error(test_labels, test_targets)
    rmse = np.sqrt(mean_squared_error(test_labels, test_targets))

    features = torch.tensor(test_features)
    labels = torch.tensor(test_labels)
    if torch.cuda.is_available():
        test_targets = model(features.to("cuda:0")).cpu().detach().numpy().flatten()
    else:
        test_targets = model(features).detach().numpy().flatten()
    test_labels = labels.cpu().detach().numpy()

    print(test_targets.flatten().shape)
    print(test_labels.shape)

    corr_coef = np.ma.corrcoef(test_labels, test_targets)[0, 1]
    mae = mean_absolute_error(test_labels, test_targets)
    mse = mean_squared_error(test_labels, test_targets)
    rmse = np.sqrt(mean_squared_error(test_labels, test_targets))

    print("correlation coefficient:")
    print(corr_coef)
    print("mae:")
    print(mae)
    print("mse:")
    print(mse)
    print("rmse:")
    print(rmse)
    """
    plt.scatter(test_labels, test_targets)
    plt.xlabel("Labels")
    plt.ylabel("Outputs")
    plt.title("DNN model with no constraints")
    plt.savefig(output_path)
    """
    with open(output_path.replace(".png", ".txt"), 'w') as file:
        file.write("Correlation coefficient: " + str(corr_coef) + "\n")
        file.write("MAE: " + str(mae) + "\n")
        file.write("MSE: " + str(mse) + "\n")
        file.write("RMSE: " + str(rmse) + "\n")
    return test_labels, test_targets        
        
def evaluate_all_predictions(model, features, labels, output_path):
    if torch.cuda.is_available():
        all_targets = model(features.to("cuda:0")).cpu().detach().numpy().flatten()
    else:
        all_targets = model(features).detach().numpy().flatten()
    all_labels = labels.cpu().detach().numpy()

    corr_coef = np.ma.corrcoef(all_labels, all_targets)[0, 1]
    mae = mean_absolute_error(all_labels, all_targets)
    mse = mean_squared_error(all_labels, all_targets)
    rmse = np.sqrt(mean_squared_error(all_labels, all_targets))

    np.save(output_path + "_labels_DO", all_labels)
    np.save(output_path + "_targets_DO", all_targets)

    print("correlation coefficient:")
    print(corr_coef)
    print("mae:")
    print(mae)
    print("mse:")
    print(mse)
    print("rmse:")
    print(rmse)

    line = np.linspace(0, np.max(all_labels), 100)
    """
    plt.scatter(all_labels, all_targets)
    plt.plot(line, line, 'r')
    plt.xlabel("Labels")
    plt.ylabel("Outputs")
    plt.title("DNN model with no constraints")
    plt.savefig(output_path + ".png")
    """
    with open(output_path + ".txt", 'w') as file:
        file.write("Correlation coefficient: " + str(corr_coef) + "\n")
        file.write("MAE: " + str(mae) + "\n")
        file.write("MSE: " + str(mse) + "\n")
        file.write("RMSE: " + str(rmse) + "\n")

    return all_labels, all_targets