import torch
import matplotlib.pyplot as plt

def load_tensor_from_file(file_path):
    try:
        return torch.load(file_path)
    except Exception as e:
        print(f"An error occurred while loading the tensor: {e}")
        return None

def plot_values(values, title, y_label, file_name):
    plt.figure()
    plt.plot(values, marker='o', linestyle='-', color='b')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(y_label)
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Load the tens
    loss_values = load_tensor_from_file('savedVars/LossPerEp.pt')
    miou_values = load_tensor_from_file('savedVars/miouPerEp.pt')

    if loss_values is not None and miou_values is not None:
        # Convert tensors to lists for plotting
        loss_values_list = loss_values
        miou_values_list = miou_values

        # Create the plots
        plot_values(loss_values_list, 'Training Loss Over Time', 'Loss', 'loss_plot.png')
        plot_values(miou_values_list, 'Validation mIoU Over Time', 'mIoU', 'miou_plot.png')
