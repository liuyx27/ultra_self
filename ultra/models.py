import torch
from torch import nn

from . import tasks, layers
from ultra.base_nbfnet import BaseNBFNet

class Ultra(nn.Module):

    def __init__(self, rel_model_cfg, entity_model_cfg):
        # kept that because super Ultra sounds cool
        super(Ultra, self).__init__()

        # adding a bit more flexibility to initializing proper rel/ent classes from the configs
        self.relation_model = globals()[rel_model_cfg.pop('class')](**rel_model_cfg)
        self.entity_model = globals()[entity_model_cfg.pop('class')](**entity_model_cfg)

        
    def forward(self, data, batch):
        
        # batch shape: (bs 批次大小, 1+num_negs 每个正样本对应的负样本数量, 3)
        # relations are the same all positive and negative triples, so we can extract only one from the first triple among 1+nug_negs
        query_rels = batch[:, 0, 2]  #query_rels一维张量，shape(bs,)，批次中每个样本的查询关系ID
        relation_representations = self.relation_model(data.relation_graph, query=query_rels)
        score = self.entity_model(data, relation_representations, batch)
        
        return score


# NBFNet to work on the graph of relations with 4 fundamental interactions
# Doesn't have the final projection MLP from hidden dim -> 1, returns all node representations 
# of shape [bs, num_rel, hidden]
class RelNBFNet(BaseNBFNet):

    #input_dim: 64   hidden_dims: [64, 64, 64, 64, 64, 64]
    #self.dims = [input_dim] + list(hidden_dims)   [64，64, 64, 64, 64, 64, 64]
    def __init__(self, input_dim, hidden_dims, num_relation=4, **kwargs):
        super().__init__(input_dim, hidden_dims, num_relation, **kwargs)

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1): #range(6)  6层GNN
            self.layers.append(
                layers.GeneralizedRelationalConv(
                    self.dims[i], self.dims[i + 1], num_relation,  
                    self.dims[0], self.message_func, self.aggregate_func, self.layer_norm,  #self.dims[0]是图卷积函数中的query_input_dim
                    self.activation, dependent=False)
                )

        if self.concat_hidden:  #默认为False
            feature_dim = sum(hidden_dims) + input_dim  #拼接所有隐藏层维度feature_dim = (64 + 64 + 64 + 64 + 64 + 64) + 64=448
            self.mlp = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(),
                nn.Linear(feature_dim, input_dim)
            )

    '''
    def bellmanford(self, data, h_index, separate_grad=False):  #h_index=query shape(bs,)
        batch_size = len(h_index)

        # initialize initial nodes (relations of interest in the batcj) with all ones
        #h_index.shape[0]=bs  query.shape (bs,64)
        query = torch.ones(h_index.shape[0], self.dims[0], device=h_index.device, dtype=torch.float)
        index = h_index.unsqueeze(-1).expand_as(query) #h_index最后一维上增加一维，h_index.shape (bs,1)

        # initial (boundary) condition - initialize all node states as zeros
        #boundary.shape (bs,关系图节点数，64)，把所有节点初始化为0
        boundary = torch.zeros(batch_size, data.num_nodes, self.dims[0], device=h_index.device)
        #boundary = torch.zeros(data.num_nodes, *query.shape, device=h_index.device)
        # Indicator function: by the scatter operation we put ones as init features of source (index) nodes
        #unsqueeze(1)之后index.shape=(bs,1,1)  query.shape=(bs,1,64)
        #scatter_add_在第二维上，将 query 张量的值添加到boundary中index 张量指定的节点位置，如果 index 对应某个位置的节点，query的特征值就会加到boundary的该位置节点的特征值上
        #boundary 的最终形状仍然是 (batch_size, num_nodes, input_dim)，但是查询特征（query）会根据 index 指定的节点位置添加到 boundary 中，
        boundary.scatter_add_(1, index.unsqueeze(1), query.unsqueeze(1))
        size = (data.num_nodes, data.num_nodes)
        edge_weight = torch.ones(data.num_edges, device=h_index.device)

        hiddens = []
        edge_weights = []
        layer_input = boundary  #shape (batch_size, num_nodes, 64)

        for layer in self.layers:
            # Bellman-Ford iteration, we send the original boundary condition in addition to the updated node states
            hidden = layer(layer_input, query, boundary, data.edge_index, data.edge_type, size, edge_weight)
            if self.short_cut and hidden.shape == layer_input.shape:
                # residual connection here
                hidden = hidden + layer_input
            hiddens.append(hidden)
            edge_weights.append(edge_weight)
            layer_input = hidden

        # original query (relation type) embeddings
        #query被复制到每个节点上，形成每个节点的查询向量
        node_query = query.unsqueeze(1).expand(-1, data.num_nodes, -1) # (batch_size, num_nodes,64)
        if self.concat_hidden:
            output = torch.cat(hiddens + [node_query], dim=-1)
            output = self.mlp(output)
        else:
            output = hiddens[-1]

        return {
            "node_feature": output,
            "edge_weights": edge_weights,
        }
    '''

    '''
    改进 为自环边增加权重 2025/1/23

    '''
    
    def bellmanford(self, data, h_index, separate_grad=False):  #h_index=query shape(bs,)
        batch_size = len(h_index)

        #初始化查询向量
        query = torch.ones(h_index.shape[0], self.dims[0], device=h_index.device, dtype=torch.float) #h_index.shape[0]=bs  query.shape (bs,64)
        index = h_index.unsqueeze(-1).expand_as(query) #h_index最后一维上增加一维，h_index.shape (bs,1)

        #初始化边界条件
        boundary = torch.zeros(batch_size, data.num_nodes, self.dims[0], device=h_index.device)#boundary.shape (bs,关系图节点数，64)，把所有节点初始化为0
        # Indicator function: by the scatter operation we put ones as init features of source (index) nodes
        #unsqueeze(1)之后index.shape=(bs,1,1)  query.shape=(bs,1,64); scatter_add_在第二维上，将query值添加到boundary中index指定的节点位置
        #boundary最终形状仍然是 (batch_size, num_nodes, input_dim)，但是查询特征query会根据index指定的节点位置添加到 boundary 中，
        boundary.scatter_add_(1, index.unsqueeze(1), query.unsqueeze(1))

        # 获取自环边的索引
        self_loops = data.edge_index[0] == data.edge_index[1]
        #num_self_loops = self_loops.sum().item()
        #print(f"自环边的数量: {num_self_loops}")

        size = (data.num_nodes, data.num_nodes)
        edge_weight = torch.ones(data.num_edges, device=h_index.device)
        # 给自环边设置不同的权重
        self_loop_weight = torch.tensor(0.3, device=h_index.device)  # 可调节的自环权重
        edge_weight[self_loops] = self_loop_weight
        edge_weight[~self_loops] = 1 - self_loop_weight

        hiddens = []
        edge_weights = []
        layer_input = boundary  #shape (batch_size, num_nodes, 64)

        for layer in self.layers:
            # Bellman-Ford iteration, we send the original boundary condition in addition to the updated node states
            hidden = layer(layer_input, query, boundary, data.edge_index, data.edge_type, size, edge_weight)
            if self.short_cut and hidden.shape == layer_input.shape:
                # residual connection here
                hidden = hidden + layer_input
            hiddens.append(hidden)
            edge_weights.append(edge_weight)
            layer_input = hidden

        # original query (relation type) embeddings
        #query被复制到每个节点上，形成每个节点的查询向量
        node_query = query.unsqueeze(1).expand(-1, data.num_nodes, -1) # (batch_size, num_nodes,64)
        if self.concat_hidden:
            output = torch.cat(hiddens + [node_query], dim=-1)
            output = self.mlp(output)
        else:
            output = hiddens[-1]

        return {
            "node_feature": output,
            "edge_weights": edge_weights,
        }

    

    def forward(self, rel_graph, query):

        # message passing and updated node representations (that are in fact relations)
        output = self.bellmanford(rel_graph, h_index=query)["node_feature"]  # (batch_size, num_nodes, hidden_dim）
        
        return output
    

class EntityNBFNet(BaseNBFNet):

    def __init__(self, input_dim, hidden_dims, num_relation=1, **kwargs):

        # dummy num_relation = 1 as we won't use it in the NBFNet layer
        super().__init__(input_dim, hidden_dims, num_relation, **kwargs)

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(
                layers.GeneralizedRelationalConv(
                    self.dims[i], self.dims[i + 1], num_relation,
                    self.dims[0], self.message_func, self.aggregate_func, self.layer_norm,
                    self.activation, dependent=False, project_relations=True)
            )

        feature_dim = (sum(hidden_dims) if self.concat_hidden else hidden_dims[-1]) + input_dim
        self.mlp = nn.Sequential()
        mlp = []
        for i in range(self.num_mlp_layers - 1):  #num_mlp_layers=2
            mlp.append(nn.Linear(feature_dim, feature_dim))
            mlp.append(nn.ReLU())
        mlp.append(nn.Linear(feature_dim, 1))
        self.mlp = nn.Sequential(*mlp)

    
    def bellmanford(self, data, h_index, r_index, separate_grad=False):  
        batch_size = len(r_index)  #长度为bs

        # initialize queries (relation types of the given triples)
        query = self.query[torch.arange(batch_size, device=r_index.device), r_index]
        index = h_index.unsqueeze(-1).expand_as(query)

        # initial (boundary) condition - initialize all node states as zeros
        boundary = torch.zeros(batch_size, data.num_nodes, self.dims[0], device=h_index.device)
        # by the scatter operation we put query (relation) embeddings as init features of source (index) nodes
        boundary.scatter_add_(1, index.unsqueeze(1), query.unsqueeze(1))
        
        size = (data.num_nodes, data.num_nodes)
        edge_weight = torch.ones(data.num_edges, device=h_index.device)

        hiddens = []
        edge_weights = []
        layer_input = boundary

        for layer in self.layers:  #6层GNN

            # for visualization
            if separate_grad:
                edge_weight = edge_weight.clone().requires_grad_()

            # Bellman-Ford iteration, we send the original boundary condition in addition to the updated node states
            hidden = layer(layer_input, query, boundary, data.edge_index, data.edge_type, size, edge_weight)
            if self.short_cut and hidden.shape == layer_input.shape:
                # residual connection here
                hidden = hidden + layer_input
            hiddens.append(hidden)
            edge_weights.append(edge_weight)
            layer_input = hidden
        #循环结束后获取每个节点的表示
            
        # original query (relation type) embeddings
        node_query = query.unsqueeze(1).expand(-1, data.num_nodes, -1) # (batch_size, num_nodes, input_dim)
        if self.concat_hidden:
            output = torch.cat(hiddens + [node_query], dim=-1)
        else:
            output = torch.cat([hiddens[-1], node_query], dim=-1)

        return {
            "node_feature": output,
            "edge_weights": edge_weights,
        }

    def forward(self, data, relation_representations, batch):  #batch.shape=(bs, 1+num_negs, 3)
        h_index, t_index, r_index = batch.unbind(-1)  #h_index.shape=(bs,1+negs)

        # initial query representations are those from the relation graph
        self.query = relation_representations

        # initialize relations in each NBFNet layer (with uinque projection internally)
        for layer in self.layers:
            layer.relation = relation_representations  #每一层的关系都来自前面计算的关系表示

        if self.training:
            # Edge dropout in the training mode
            # here we want to remove immediate edges直接边 (head, relation, tail) from the edge_index and edge_types
            # to make NBFNet iteration learn non-trivial paths迭代学习非平凡路径
            data = self.remove_easy_edges(data, h_index, t_index, r_index)

        shape = h_index.shape  #h_index.shape=(bs,1+negs)
        # turn all triples in a batch into a tail prediction mode
        h_index, t_index, r_index = self.negative_sample_to_tail(h_index, t_index, r_index, num_direct_rel=data.num_relations // 2)
        assert (h_index[:, [0]] == h_index).all()
        assert (r_index[:, [0]] == r_index).all()  #确保负采样之后头节点和关系没有变

        # message passing and updated node representations    h_index[:, 0].shape=(bs,)
        output = self.bellmanford(data, h_index[:, 0], r_index[:, 0])  # ( batch_size, num_nodes,feature_dim）
        feature = output["node_feature"]
        index = t_index.unsqueeze(-1).expand(-1, -1, feature.shape[-1])  #shape=(bs,1+negs,feature_dim) t_index的最后添加一个维度，扩展到与feature.shape[-1] 相同的维度
        # extract representations of tail entities from the updated node states
        #gather(1, index) 从feature中根据index提取特定元素，提取尾节点特征
        feature = feature.gather(1, index)  # (batch_size, num_negative + 1, feature_dim)

        # probability logit for each tail node in the batch
        # (batch_size, num_negative + 1, dim) -> (batch_size, num_negative + 1)
        score = self.mlp(feature).squeeze(-1)
        return score.view(shape)


class QueryNBFNet(EntityNBFNet): 
    """
    The entity-level reasoner for UltraQuery-like complex query answering pipelines
    Almost the same as EntityNBFNet except that 
    (1) we already get the initial node features at the forward pass time 
    and don't have to read the triples batch
    (2) we get `query` from the outer loop
    (3) we return a distribution over all nodes (assuming t_index = all nodes)
    """
    
    def bellmanford(self, data, node_features, query, separate_grad=False):
        
        size = (data.num_nodes, data.num_nodes)
        edge_weight = torch.ones(data.num_edges, device=query.device)

        hiddens = []
        edge_weights = []
        layer_input = node_features

        for layer in self.layers:

            # for visualization
            if separate_grad:
                edge_weight = edge_weight.clone().requires_grad_()

            # Bellman-Ford iteration, we send the original boundary condition in addition to the updated node states
            hidden = layer(layer_input, query, node_features, data.edge_index, data.edge_type, size, edge_weight)
            if self.short_cut and hidden.shape == layer_input.shape:
                # residual connection here
                hidden = hidden + layer_input
            hiddens.append(hidden)
            edge_weights.append(edge_weight)
            layer_input = hidden

        # original query (relation type) embeddings
        node_query = query.unsqueeze(1).expand(-1, data.num_nodes, -1) # (batch_size, num_nodes, input_dim)
        if self.concat_hidden:
            output = torch.cat(hiddens + [node_query], dim=-1)
        else:
            output = torch.cat([hiddens[-1], node_query], dim=-1)

        return {
            "node_feature": output,
            "edge_weights": edge_weights,
        }

    def forward(self, data, node_features, relation_representations, query):

        # initialize relations in each NBFNet layer (with uinque projection internally)
        for layer in self.layers:
            layer.relation = relation_representations

        # we already did traversal_dropout in the outer loop of UltraQuery
        # if self.training:
        #     # Edge dropout in the training mode
        #     # here we want to remove immediate edges (head, relation, tail) from the edge_index and edge_types
        #     # to make NBFNet iteration learn non-trivial paths
        #     data = self.remove_easy_edges(data, h_index, t_index, r_index)

        # node features arrive in shape (bs, num_nodes, dim)
        # NBFNet needs batch size on the first place
        output = self.bellmanford(data, node_features, query)  # (num_nodes, batch_size, feature_dim）
        score = self.mlp(output["node_feature"]).squeeze(-1) # (bs, num_nodes)
        return score  

    


