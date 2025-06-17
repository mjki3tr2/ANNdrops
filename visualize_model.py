def visualize_model(model,filename=None,input_labels=None,output_labels=None):
    import VisualizeNN as VisNN
    import numpy as np
    
    network_structure = [model.input_shape[1]]
    weights = []

    for layer in model.layers:
        if 'dense' in layer.name:
            network_structure.append(layer.units)
            weights.append(layer.get_weights()[0])

    network = VisNN.DrawNN(np.array(network_structure), weights)
    network.draw(filename=filename, ilabels=input_labels, olabels=output_labels)
    return network