import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

EMB_INIT_EPS = 2.0
gamma = 12.0


class cmafddiModel(nn.Module):

    def __init__(self, args, entity_pre_embed=None, structure_pre_embed=None, protein_pre_embed=None,smiles_pre_embed=None):

        super(cmafddiModel, self).__init__()

        self.fusion_type = args.feature_fusion

        # embedding setting
        self.structure_dim = args.structure_dim
        self.entity_dim = args.entity_dim
        self.protein_dim = args.protein_dim
        # embedding data
        self.structure_pre_embed = structure_pre_embed
        self.entity_pre_embed = entity_pre_embed
        self.protein_pre_embed = protein_pre_embed
        self.smiles_pre_embed = smiles_pre_embed
        self.n_approved_drug = structure_pre_embed.shape[0]

        # self.n_entities = n_entities
        # self.n_relations = n_relations

        self.multi_type = args.multi_type

        # self.conv_dim_list = [args.entity_dim] + eval(args.conv_dim_list)
        # self.mess_dropout = eval(args.mess_dropout)
        # self.n_layers = len(eval(args.conv_dim_list))

        # self.ddi_l2loss_lambda = args.DDI_l2loss_lambda

        self.hidden_dim = args.entity_dim
        self.eps = EMB_INIT_EPS
        self.emb_init = (gamma + self.eps) / self.hidden_dim

        # fusion type
        if self.fusion_type == 'concat':

            self.layer1_f = nn.Sequential(
                nn.Linear(self.structure_dim + self.entity_dim + self.protein_dim, self.entity_dim),
                nn.BatchNorm1d(self.entity_dim),
                nn.LeakyReLU(True))
            self.layer2_f = nn.Sequential(nn.Linear(self.entity_dim, self.entity_dim),
                                          nn.BatchNorm1d(self.entity_dim),
                                          nn.LeakyReLU(True))
            self.layer3_f = nn.Sequential(nn.Linear(self.entity_dim, self.entity_dim),
                                          nn.BatchNorm1d(self.entity_dim),
                                          nn.LeakyReLU(True))

        elif self.fusion_type == 'sum':

            self.W_s = nn.Linear(self.structure_dim, self.entity_dim)
            self.W_e = nn.Linear(self.entity_dim, self.entity_dim)
            self.W_p = nn.Linear(self.protein_dim, self.entity_dim)


        elif self.fusion_type == 'CMAF':
            self.druglayer_structure = nn.Linear(self.structure_dim, self.entity_dim)
            self.druglayer_KG = nn.Linear(self.entity_dim, self.entity_dim)
            self.druglayer_protein = nn.Linear(self.protein_dim, self.entity_dim)
            self.attention = nn.MultiheadAttention(embed_dim=720, num_heads=3)
            self.attention_layer = nn.Sequential(
                nn.Linear(820, 2460),
                nn.ReLU(),
                nn.Linear(2460, 2460)
            )
            encoder_layers = TransformerEncoderLayer(d_model=100, nhead=4)
            self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=2)
            self.multi_drug = nn.Sequential(nn.Linear(self.entity_dim, self.entity_dim))
            self.activate = nn.ReLU()


        if self.fusion_type in ['concat']:
            self.all_embedding_dim = self.entity_dim*2
        elif self.fusion_type in ['CMAF']:
            self.all_embedding_dim = (self.structure_dim + self.entity_dim*2+ self.protein_dim) *6
        elif self.fusion_type in ['sum']:
            self.all_embedding_dim = self.entity_dim*2


        self.layer1 = nn.Sequential(nn.Linear(self.all_embedding_dim, args.n_hidden_1), nn.BatchNorm1d(args.n_hidden_1),
                                    nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(args.n_hidden_1, args.n_hidden_2), nn.BatchNorm1d(args.n_hidden_2),
                                    nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(args.n_hidden_2, args.out_dim))

    def generate_fusion_feature(self, train_data, batch_data):
        # we focus on approved drug
        global embedding_data
        global embedding_data_reverse

        self.entity_embed_pre = self.entity_pre_embed[:1706, :]


        if self.fusion_type == 'concat':

            x = torch.cat([self.structure_pre_embed, self.entity_embed_pre, self.protein_pre_embed], dim=1)
            x = self.layer1_f(x)
            x = self.layer2_f(x)
            x = self.layer3_f(x)

            return x

        elif self.fusion_type == 'sum':
            structure = self.W_s(self.structure_pre_embed)
            entity = self.W_e(self.entity_embed_pre)
            protein = self.W_p(self.protein_pre_embed)
            all = structure + entity + protein

            return all


        elif self.fusion_type == 'CMAF':
            structure = self.druglayer_structure(self.structure_pre_embed)

            entity = self.druglayer_KG(self.entity_embed_pre)

            protein = self.druglayer_protein(self.protein_pre_embed)

            out3 = self.activate(self.multi_drug(structure * entity * protein))

            concatenated = torch.cat([self.structure_pre_embed, self.entity_embed_pre, self.protein_pre_embed], dim=1)
            attn_output, _ = self.attention(concatenated.unsqueeze(0), concatenated.unsqueeze(0),
                                            concatenated.unsqueeze(0))
            result = torch.cat((attn_output.squeeze(0), out3,), dim=1)
            fused_feature = self.attention_layer(result)


            return fused_feature

        elif self.fusion_type == 'sum':

            #structure = self.W_s(self.structure_pre_embed)
            protein=self.W_s(self.protein_pre_embed)
            entity = self.W_e(self.entity_embed_pre)
            add_structure_entity = protein + entity

            return add_structure_entity

    def train_DDI_data(self, train_data, batch_data):

        # all_embed = self.generate_fusion_feature(batch_data)

        drug1_embed = self.all_embed[train_data[:, 0]]
        drug2_embed = self.all_embed[train_data[:, 1]]
        drug_data = torch.cat((drug1_embed, drug2_embed), 1)

        x = self.layer1(drug_data)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

    def test_DDI_data(self, test_data, batch_data):

        # all_embed = self.generate_fusion_feature(batch_data)
        drug1_embed = self.all_embed[test_data[:, 0]]
        drug2_embed = self.all_embed[test_data[:, 1]]
        drug_data = torch.cat((drug1_embed, drug2_embed), 1)

        x = self.layer1(drug_data)
        x = self.layer2(x)
        x = self.layer3(x)

        if self.multi_type != 'False':
            pred = F.softmax(x, dim=1)
        else:
            pred = torch.sigmoid(x)

        return pred, self.all_embed

    def forward(self, mode, *input):
        self.all_embed = self.generate_fusion_feature(*input)
        if mode == 'calc_ddi_loss':
            return self.train_DDI_data(*input)
        if mode == 'predict':
            return self.test_DDI_data(*input)

