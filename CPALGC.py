import torch
import numpy as np
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType
import scipy.sparse as sp

from PairNorm import PairNorm


class CPALGC(GeneralRecommender):
    r"""
    We implement CPA-LGC following the implementation of LightGCN.
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset, n_cri, edge_weights):
        super(CPALGC, self).__init__(config, dataset)

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(
            form='coo').astype(np.float32)
        # should be replaced with variable later
        self.cri_idx_shift = int((self.n_items)/n_cri)
        self.n_cri = n_cri
        self.edge_weights = edge_weights

        # load parameters info
        # int type:the embedding size of lightGCN
        self.latent_dim = config['embedding_size']
        # int type:the layer num of lightGCN
        self.n_layers = config['n_layers']
        # float32 type: the weight decay for l2 normalization
        self.reg_weight = config['reg_weight']

        # define layers and loss
        self.user_embedding = torch.nn.Embedding(
            num_embeddings=self.n_users, embedding_dim=self.latent_dim)
        self.item_embedding = torch.nn.Embedding(
            num_embeddings=self.n_items, embedding_dim=self.latent_dim)
        self.cri_user_embedding = torch.nn.Embedding(
            num_embeddings=self.n_users, embedding_dim=self.latent_dim)
        self.cri_embedding_item = torch.zeros(
            (self.n_items, self.latent_dim), device=config['device'])

        self.norm = PairNorm('PN', scale=1)

        self.__init_criteria_weight()

        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # generate intermediate data
        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)

        # parameters initialization
        self.apply(xavier_uniform_initialization)

    def __init_criteria_weight(self):

        with torch.no_grad():
            self.cri_emb_nograd = torch.zeros((self.n_cri, self.latent_dim))

        # nn.init.normal(self.cri_emb_nograd, std=0.1)
        torch.nn.init.xavier_uniform_(self.cri_emb_nograd)

        self.cri_emb_nograd[0] = torch.zeros(self.latent_dim)

        for i in range(self.n_items):
            if i // self.cri_idx_shift == 0:
                pass
            elif i // self.cri_idx_shift >= self.n_cri:
                IndexError
            else:
                self.cri_embedding_item[i] = self.cri_emb_nograd[i //
                                                                 self.cri_idx_shift]

    def get_norm_adj_mat(self):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        # build adj matrix
        A = sp.dok_matrix((self.n_users + self.n_items,
                          self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(
            zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(
            dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        A._update(data_dict)

        # Element-wise multiplication with adjacency matrix and edge weights
        A = A.multiply(self.edge_weights)

        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor([row, col])
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.

        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)

        cri_user_embeddings = self.cri_user_embedding.weight
        cri_item_embeddings = self.cri_embedding_item
        cri_ego_embeddings = torch.cat(
            [cri_user_embeddings, cri_item_embeddings], dim=0)
        return ego_embeddings, cri_ego_embeddings

    def forward(self):
        all_embeddings, cri_all_embeddings = self.get_ego_embeddings()

        all_embeddings = self.norm(all_embeddings)
        cri_all_embeddings = self.norm(cri_all_embeddings)
        embeddings_list = [all_embeddings]
        cri_embeddings_list = [cri_all_embeddings]

        for layer_idx in range(self.n_layers):
            all_embeddings = torch.sparse.mm(
                self.norm_adj_matrix, all_embeddings)
            all_embeddings = self.norm(all_embeddings)
            embeddings_list.append(all_embeddings)

        for layer_idx in range(self.n_layers):
            cri_all_embeddings = torch.sparse.mm(
                self.norm_adj_matrix, cri_all_embeddings)
            cri_all_embeddings = self.norm(cri_all_embeddings)
            cri_embeddings_list.append(cri_all_embeddings)

        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        lightgcn_cri_embeddings = torch.stack(cri_embeddings_list, dim=1)
        lightgcn_cri_embeddings = torch.mean(lightgcn_cri_embeddings, dim=1)

        lightgcn_all_embeddings = self.norm(lightgcn_all_embeddings)
        lightgcn_cri_embeddings = self.norm(lightgcn_cri_embeddings)

        # concat-version
        lightgcn_all_embeddings = torch.cat(
            [lightgcn_all_embeddings, lightgcn_cri_embeddings], dim=1)

        # sum-version
        # lightgcn_all_embeddings = lightgcn_all_embeddings + lightgcn_cri_embeddings

        user_all_embeddings, item_all_embeddings = torch.split(
            lightgcn_all_embeddings, [self.n_users, self.n_items])

        return user_all_embeddings, item_all_embeddings

    def calculate_loss(self, interaction):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        # calculate BPR Loss
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        # calculate reg Loss
        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)

        reg_loss = self.reg_loss(
            u_ego_embeddings, pos_ego_embeddings, neg_ego_embeddings)
        loss = mf_loss + self.reg_weight * reg_loss

        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()
        # get user embedding from storage variable
        u_embeddings = self.restore_user_e[user]
        # dot with all item embedding to accelerate
        scores = torch.matmul(
            u_embeddings, self.restore_item_e.transpose(0, 1))
        # We only consider overall interactions.
        scores[:, int(self.n_items/self.n_cri):] = -np.inf

        return scores.view(-1)
