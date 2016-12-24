#!/usr/bin/env python


class Neuron:
    """Model of neuron in neural network.

    Attributes:

        `inbound_neurons`: Neuron from which this neuron receives the values.
        `outbound_variables`: Neuron to which this neuron pass values.
        `value`: Calculated value.

    """
    def __init__(self, inbound_neurons=[]):
        """Add this node as an outbound node on its inputs.

        Arguments:

            `inbound_neurons`: Neuron from which this neuron receives the values.  # NOQA

        """
        self.inbound_neurons = inbound_neurons
        self.outbound_neurons = []

        for i in inbound_neurons:
            i.outbound_neurons.append(self)

        self.value = None

    def forward(self):
        """Forward propagation.

        Compute the output base on `inbound_neurons` and
        store the result in self.value.
        """
        raise NotImplementedError

    def backward(self):
        """Backward propagation.

        Compute later
        """
        raise NotImplementedError


class Input(Neuron):
    """Model of input neuron.
    """
    def __init__(self):
        # An Input neuron has no inbound neurons,
        # so no need to pass anything to the Neuron instantiator.
        Neuron.__init__(self)

    def forward(self, value=None):
        """Overwrite the value if one is passed in
        """
        if value is not None:
            self.value = value


class Add(Neuron):
    """Perform a addtion
    """
    def __init__(self, *inbound_neurons):
        Neuron.__init__(self, inbound_neurons)

    def forward(self):
        """Set the value of this neuron (`self.value`) to the sum of it's inbound_nodes.  # NOQA
        """
        self.value = 0
        for neuron in self.inbound_neurons:
            self.value += neuron.value


def topological_sort(feed_dict):
    """
    Sort the neurons in topological order using Kahn's Algorithm.

    `feed_dict`: A dictionary where the key is a `Input` Neuron and the value is the respective value feed to that Neuron.  # NOQA

    Returns a list of sorted neurons.
    """

    input_neurons = [n for n in feed_dict.keys()]

    G = {}
    neurons = [n for n in input_neurons]
    while len(neurons) > 0:
        n = neurons.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
        for m in n.outbound_neurons:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            neurons.append(m)

    L = []
    S = set(input_neurons)
    while len(S) > 0:
        n = S.pop()

        if isinstance(n, Input):
            n.value = feed_dict[n]

        L.append(n)
        for m in n.outbound_neurons:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            # if no other incoming edges add to S
            if len(G[m]['in']) == 0:
                S.add(m)
    return L


def forward_pass(output_Neuron, sorted_neurons):
    """
    Performs a forward pass through a list of sorted neurons.

    Arguments:

        `output_Neuron`: A Neuron in the graph, should be the output Neuron (have no outgoing edges).  # NOQA
        `sorted_neurons`: a topologically sorted list of neurons.

    Returns the output Neuron's value
    """

    for n in sorted_neurons:
        n.forward()

    return output_Neuron.value
