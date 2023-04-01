import ml_collections

def fcn_potential(data_dim):
    model = ml_collections.ConfigDict()
    model.name = 'fcn_potential'
    model.sigmoid_last = False
    model.state_size = data_dim
    model.hidden_layers = 3
    model.hidden_nodes = 256
    model.dropout = 0.0

    return model