def visualize_model(model,filename=None,input_labels=None,output_labels=None):
    import VisualizeNN as VisNN
    import numpy as np
    
    network_structure = [model.input_shape[1]]
    weights = []

    for layer in model.layers:
        if 'Dense' in layer.name or 'Output' in layer.name:
            network_structure.append(layer.units)
            weights.append(layer.get_weights()[0])
    
    normalized_weights = []
    for w in weights:
        max_abs = np.max(np.abs(w))
        if max_abs == 0:
            max_abs = 1.0
        normalized_weights.append(w / max_abs)
    
    #all_weights = np.concatenate([w.flatten() for w in weights])
    #max_abs = np.max(np.abs(all_weights))
    #if max_abs == 0:
    #    max_abs = 1.0
    #normalized_weights = [w / max_abs for w in weights]
    
    network = VisNN.DrawNN(np.array(network_structure), normalized_weights)
    network.draw(filename=filename, ilabels=input_labels, olabels=output_labels)
    return network, weights