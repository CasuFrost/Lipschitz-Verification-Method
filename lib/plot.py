import matplotlib.pyplot as plt
import numpy as np
def plot_scatter_points(list_of_points):
    """
    Plots multiple sets of points in R^2, each set with a different color,
    without connecting them with lines.

    Args:
        list_of_points: A list of lists, where each sublist contains points
                        in the form of tuples or lists (x, y).
                        Example: [[(x1, y1), (x2, y2)], [(x3, y3), (x4, y4)]]
    """
    # Set of colors to use for plotting.
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    num_colors = len(colors)

    if not all(isinstance(sublist, list) for sublist in list_of_points):
        print("Input must be a list of lists.")
        return

    # Create a new figure
    plt.figure()

    # Iterate through each sublist and plot the points
    for i, point_set in enumerate(list_of_points):
        if not point_set:
            continue
        
        # Unzip the points into separate x and y lists
        x_coords, y_coords = zip(*point_set)

        # Get the color from our list, cycling through them
        color = colors[i % num_colors]

        # Plot the points with the assigned color and marker, without lines
        plt.plot(x_coords, y_coords, color=color, marker='o', linestyle='', label=f'Set di Punti {i+1}')
    
    # Add labels and a title to the plot
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Plot di punti sparsi in R^2')
    plt.legend()
    plt.grid(True)
    
    # Display the plot
    plt.show()



'''ESEGUIRE IL PLOT DELLE 3 FUNZIONI DI PERDITA COME CURVE DIFFERENTI'''
def plot_losses(path,neuron_par_layer,num_layer,packed_losses):
    plt.clf()
    x_data = []
    Z0_losses = []
    ZU_losses = []
    Dynamic_losses = []
    global_losses = []

    with open(path, 'r') as f:
        for line in f:
            s = line.split(' ')
            x_data.append(int(s[0]))
            global_losses.append(float(s[1]))
            Z0_losses.append(float(s[2]))
            ZU_losses.append(float(s[3]))
            Dynamic_losses.append(float(s[4]))
    plt.plot(x_data, global_losses, color='black',label='loss')
    plt.plot(x_data, Z0_losses,linestyle='--', color='blue',label='Z0')
    plt.plot(x_data, ZU_losses, color='red',linestyle='--',label='ZU')
    plt.plot(x_data, Dynamic_losses, color='green',linestyle='--',label='Dyn')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    avg_loss = ' NO DATA '
    if len(global_losses)>=15:
        avg_loss=str(np.mean(global_losses[-15:]))
    plt.title('Min loss : '+str(min(global_losses))+'\n'
    'Avg last 15 loss : '+avg_loss+'\n'
    'Neuron per layer:'+str(neuron_par_layer)+' Num layers : '+str(num_layer))
    plt.grid(True) # Add a grid for better readability
    plt.savefig(path[:-3]+'pdf')
    #plt.show()