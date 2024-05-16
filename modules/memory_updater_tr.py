'''
from torch import nn
import torch

class MemoryUpdater(nn.Module):
    def update_memory(self, unique_node_ids, unique_messages, timestamps, transaction_features):
        pass

class SequenceMemoryUpdater(MemoryUpdater):
    def __init__(self, memory, message_dimension, memory_dimension, transaction_feature_dimension, device):
        super(SequenceMemoryUpdater, self).__init__()
        self.memory = memory
        self.layer_norm = torch.nn.LayerNorm(memory_dimension)
        self.message_dimension = message_dimension
        self.transaction_feature_dimension = transaction_feature_dimension
        self.device = device

        # Define layers for processing transaction features if needed
        if transaction_feature_dimension > 0:
            self.transaction_feature_layer = nn.Linear(transaction_feature_dimension, memory_dimension)

        # Define layer for processing concatenated inputs
        input_dimension = message_dimension + transaction_feature_dimension
        self.concatenation_layer = nn.Linear(input_dimension, memory_dimension)

    def update_memory(self, unique_node_ids, unique_messages, timestamps, transaction_features):
        if len(unique_node_ids) <= 0:
            return

        assert (self.memory.get_last_update(unique_node_ids) <= timestamps).all().item(), "Trying to " \
                                                                                          "update memory to time in the past"

        memory = self.memory.get_memory(unique_node_ids)
        self.memory.last_update[unique_node_ids] = timestamps

        # Process transaction features if needed
        if self.transaction_feature_dimension > 0:
            transaction_features = self.transaction_feature_layer(transaction_features)

        # Concatenate message embeddings with transaction features
        concatenated_inputs = torch.cat((unique_messages, transaction_features), dim=1)

        # Pass concatenated inputs through a layer to get the final input for memory updater
        final_inputs = self.concatenation_layer(concatenated_inputs)

        updated_memory = self.memory_updater(final_inputs, memory)

        self.memory.set_memory(unique_node_ids, updated_memory)

    def get_updated_memory(self, unique_node_ids, unique_messages, timestamps, transaction_features):
        if len(unique_node_ids) <= 0:
            return self.memory.memory.data.clone(), self.memory.last_update.data.clone()

        assert (self.memory.get_last_update(unique_node_ids) <= timestamps).all().item(), "Trying to " \
                                                                                          "update memory to time in the past"

        updated_memory = self.memory.memory.data.clone()
        updated_memory[unique_node_ids] = self.memory_updater(final_inputs, updated_memory[unique_node_ids])

        updated_last_update = self.memory.last_update.data.clone()
        updated_last_update[unique_node_ids] = timestamps

        return updated_memory, updated_last_update


class GRUMemoryUpdater(SequenceMemoryUpdater):
    def __init__(self, memory, message_dimension, memory_dimension, transaction_feature_dimension, device):
        super(GRUMemoryUpdater, self).__init__(memory, message_dimension, memory_dimension, transaction_feature_dimension, device)

        self.memory_updater = nn.GRUCell(input_size=memory_dimension,
                                         hidden_size=memory_dimension)


class RNNMemoryUpdater(SequenceMemoryUpdater):
    def __init__(self, memory, message_dimension, memory_dimension, transaction_feature_dimension, device):
        super(RNNMemoryUpdater, self).__init__(memory, message_dimension, memory_dimension, transaction_feature_dimension, device)

        self.memory_updater = nn.RNNCell(input_size=memory_dimension,
                                         hidden_size=memory_dimension)

def get_memory_updater(module_type, memory, message_dimension, memory_dimension, transaction_feature_dimension, device):
    if module_type == "gru":
        return GRUMemoryUpdater(memory, message_dimension, memory_dimension, transaction_feature_dimension, device)
    elif module_type == "rnn":
        return RNNMemoryUpdater(memory, message_dimension, memory_dimension, transaction_feature_dimension, device)
'''

from torch import nn
import torch

class MemoryUpdater(nn.Module):
    def update_memory(self, unique_node_ids, unique_messages, timestamps, transaction_features):
        pass

# class SequenceMemoryUpdater(MemoryUpdater):
#     def __init__(self, memory, message_dimension, memory_dimension, transaction_feature_dimension, device):
#         super(SequenceMemoryUpdater, self).__init__()
#         self.memory = memory
#         self.layer_norm = torch.nn.LayerNorm(memory_dimension)
#         self.message_dimension = message_dimension
#         self.transaction_feature_dimension = transaction_feature_dimension
#         self.device = device

#         # Define layers for processing transaction features if needed
#         if transaction_feature_dimension > 0:
#             self.transaction_feature_layer = nn.Linear(transaction_feature_dimension, memory_dimension)

#         # Define layer for processing concatenated inputs
#         input_dimension = message_dimension + transaction_feature_dimension
#         self.concatenation_layer = nn.Linear(input_dimension, memory_dimension)

    # def update_memory(self, unique_node_ids, unique_messages, timestamps, transaction_features):
    #     if len(unique_node_ids) <= 0:
    #         return

    #     assert (self.memory.get_last_update(unique_node_ids) <= timestamps).all().item(), "Trying to " \
    #                                                                                       "update memory to time in the past"

    #     memory = self.memory.get_memory(unique_node_ids)
    #     self.memory.last_update[unique_node_ids] = timestamps

    #     # Process transaction features if needed
    #     if self.transaction_feature_dimension > 0:
    #         transaction_features = self.transaction_feature_layer(transaction_features)

    #     # Concatenate message embeddings with transaction features
    #     concatenated_inputs = torch.cat((unique_messages, transaction_features), dim=1)

    #     # Pass concatenated inputs through a layer to get the final input for memory updater
    #     final_inputs = self.concatenation_layer(concatenated_inputs)

    #     updated_memory = self.memory_updater(final_inputs, memory)

    #     self.memory.set_memory(unique_node_ids, updated_memory)

    # def get_updated_memory(self, unique_node_ids, unique_messages, timestamps, transaction_features):
    #     if len(unique_node_ids) <= 0:
    #         return self.memory.memory.data.clone(), self.memory.last_update.data.clone()

    #     assert (self.memory.get_last_update(unique_node_ids) <= timestamps).all().item(), "Trying to " \
    #                                                                                       "update memory to time in the past"

    #     updated_memory = self.memory.memory.data.clone()
    #     updated_memory[unique_node_ids] = self.memory_updater(final_inputs, updated_memory[unique_node_ids])

    #     updated_last_update = self.memory.last_update.data.clone()
    #     updated_last_update[unique_node_ids] = timestamps

    #     return updated_memory, updated_last_update

class SequenceMemoryUpdater(MemoryUpdater):
    def __init__(self, memory, message_dimension, memory_dimension, device):
        super(SequenceMemoryUpdater, self).__init__()
        self.memory = memory
        self.layer_norm = torch.nn.LayerNorm(memory_dimension)
        self.message_dimension = message_dimension
        self.device = device

        # Define layer for processing concatenated inputs
        input_dimension = message_dimension + 1  # Add 1 for net transaction amount feature
        self.concatenation_layer = nn.Linear(input_dimension, memory_dimension)

    def update_memory(self, unique_node_ids, unique_messages, timestamps, net_transaction_amounts):
        if len(unique_node_ids) <= 0:
            return
        
        assert (self.memory.get_last_update(unique_node_ids) <= timestamps).all().item(), "Trying to " \
                                                                                          "update memory to time in the past"

        memory = self.memory.get_memory(unique_node_ids)
        self.memory.last_update[unique_node_ids] = timestamps

        # Concatenate message embeddings with net transaction amount feature
        net_transaction_amounts_tensor = torch.tensor(net_transaction_amounts, dtype=torch.float32, device=self.device)
        concatenated_inputs = torch.cat((unique_messages, net_transaction_amounts_tensor.unsqueeze(1)), dim=1)

        # Pass concatenated inputs through a layer to get the final input for memory updater
        final_inputs = self.concatenation_layer(concatenated_inputs)

        updated_memory = self.memory_updater(final_inputs, memory)

        self.memory.set_memory(unique_node_ids, updated_memory)

    def get_updated_memory(self, unique_node_ids, unique_messages, timestamps, net_transaction_amounts):
        if len(unique_node_ids) <= 0:
            return self.memory.memory.data.clone(), self.memory.last_update.data.clone()

        assert (self.memory.get_last_update(unique_node_ids) <= timestamps).all().item(), "Trying to " \
                                                                                          "update memory to time in the past"

        updated_memory = self.memory.memory.data.clone()
        
        # Concatenate message embeddings with net transaction amount feature
        net_transaction_amounts_tensor = torch.tensor(net_transaction_amounts, dtype=torch.float32, device=self.device)
        concatenated_inputs = torch.cat((unique_messages, net_transaction_amounts_tensor.unsqueeze(1)), dim=1)

        # Pass concatenated inputs through a layer to get the final input for memory updater
        final_inputs = self.concatenation_layer(concatenated_inputs)

        updated_memory[unique_node_ids] = self.memory_updater(final_inputs, updated_memory[unique_node_ids])

        updated_last_update = self.memory.last_update.data.clone()
        updated_last_update[unique_node_ids] = timestamps

        return updated_memory, updated_last_update

class GRUMemoryUpdater(SequenceMemoryUpdater):
    def __init__(self, memory, message_dimension, memory_dimension, transaction_feature_dimension, device):
        super(GRUMemoryUpdater, self).__init__(memory, message_dimension, memory_dimension, transaction_feature_dimension, device)

        self.memory_updater = nn.GRUCell(input_size=memory_dimension,
                                         hidden_size=memory_dimension)


class RNNMemoryUpdater(SequenceMemoryUpdater):
    def __init__(self, memory, message_dimension, memory_dimension, transaction_feature_dimension, device):
        super(RNNMemoryUpdater, self).__init__(memory, message_dimension, memory_dimension, transaction_feature_dimension, device)

        self.memory_updater = nn.RNNCell(input_size=memory_dimension,
                                         hidden_size=memory_dimension)

def get_memory_updater(module_type, memory, message_dimension, memory_dimension, transaction_feature_dimension, device):
    if module_type == "gru":
        return GRUMemoryUpdater(memory, message_dimension, memory_dimension, transaction_feature_dimension, device)
    elif module_type == "rnn":
        return RNNMemoryUpdater(memory, message_dimension, memory_dimension, transaction_feature_dimension, device)
