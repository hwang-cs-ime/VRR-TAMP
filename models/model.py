import torch
import torch.nn as nn
from models.detr import build_detr, build_VLFusion
from pytorch_pretrained_bert.modeling import BertModel


def generate_coord(batch, height, width):
    xv, yv = torch.meshgrid([torch.arange(0, height), torch.arange(0, width)])
    xv_min = (xv.float() * 2 - width) / width
    yv_min = (yv.float() * 2 - height) / height
    xv_max = ((xv + 1).float() * 2 - width) / width
    yv_max = ((yv + 1).float() * 2 - height) / height
    xv_ctr = (xv_min + xv_max) / 2
    yv_ctr = (yv_min + yv_max) / 2
    hmap = torch.ones(height, width) * (1. / height)
    wmap = torch.ones(height, width) * (1. / width)
    coord = torch.autograd.Variable(torch.cat([xv_min.unsqueeze(0), yv_min.unsqueeze(0), \
                                               xv_max.unsqueeze(0), yv_max.unsqueeze(0), \
                                               xv_ctr.unsqueeze(0), yv_ctr.unsqueeze(0), \
                                               hmap.unsqueeze(0), wmap.unsqueeze(0)], dim=0).cuda())
    coord = coord.unsqueeze(0).repeat(batch, 1, 1, 1)
    return coord


def load_weights(model, load_path):
    dict_trained = torch.load(load_path)['model']
    dict_new = model.state_dict().copy()
    for key in dict_new.keys():
        if key in dict_trained.keys():
            dict_new[key] = dict_trained[key]
    model.load_state_dict(dict_new)
    del dict_new
    del dict_trained
    torch.cuda.empty_cache()
    return model


class VRR_TAMP(nn.Module):
    def __init__(self, bert_model='bert-base-uncased', tunebert=True, args=None):
        super(VRR_TAMP, self).__init__()
        self.tunebert = tunebert
        if bert_model == 'bert-base-uncased':
            self.textdim = 768
        else:
            self.textdim = 1024

        # Visual model
        self.visumodel = build_detr(args)
        self.visumodel = load_weights(self.visumodel, './saved_models/detr-r50-e632da11.pth')

        # Text model
        self.textmodel = BertModel.from_pretrained(bert_model)

        # Visual-linguistic Fusion model
        self.vlmodel = build_VLFusion(args)
        self.vlmodel = load_weights(self.vlmodel, './saved_models/detr-r50-e632da11.pth')

        # Subject
        self.f_subject = torch.nn.Sequential(nn.Linear(256, 256), nn.ReLU(), )
        # gate unit for subject
        self.g_subject = torch.nn.Sequential(nn.Linear(256, 256), nn.ReLU(), )
        # message passing -> object
        self.m_s2o = torch.nn.Sequential(nn.Linear(256, 256), nn.ReLU(), )
        # final output for subject
        self.output_subject = torch.nn.Sequential(nn.Linear(256, 4), )

        for p_f_s in self.f_subject.parameters():
            if p_f_s.dim() > 1:
                nn.init.xavier_uniform_(p_f_s)

        for p_g_s in self.g_subject.parameters():
            if p_g_s.dim() > 1:
                nn.init.xavier_uniform_(p_g_s)

        for p_m_s2o in self.m_s2o.parameters():
            if p_m_s2o.dim() > 1:
                nn.init.xavier_uniform_(p_m_s2o)

        for p_output_s in self.output_subject.parameters():
            if p_output_s.dim() > 1:
                nn.init.xavier_uniform_(p_output_s)

        # Object
        self.f_object = torch.nn.Sequential(nn.Linear(256, 256), nn.ReLU(), )
        # gate unit for object
        self.g_object = torch.nn.Sequential(nn.Linear(256, 256), nn.ReLU(), )
        # message passing -> subject
        self.m_o2s = torch.nn.Sequential(nn.Linear(256, 256), nn.ReLU(), )
        # final output for object
        self.output_object = torch.nn.Sequential(nn.Linear(256, 4), )

        for p_f_o in self.f_object.parameters():
            if p_f_o.dim() > 1:
                nn.init.xavier_uniform_(p_f_o)

        for p_g_o in self.g_object.parameters():
            if p_g_o.dim() > 1:
                nn.init.xavier_uniform_(p_g_o)

        for p_m_o2s in self.m_o2s.parameters():
            if p_m_o2s.dim() > 1:
                nn.init.xavier_uniform_(p_m_o2s)

        for p_output_o in self.output_object.parameters():
            if p_output_o.dim() > 1:
                nn.init.xavier_uniform_(p_output_o)

    def forward(self, image, mask, word_id, word_mask):
        # Visual Module
        fv = self.visumodel(image, mask)

        # Language Module
        all_encoder_layers, _ = self.textmodel(word_id, token_type_ids=None, attention_mask=word_mask)

        fl = (all_encoder_layers[-1] + all_encoder_layers[-2] + all_encoder_layers[-3] + all_encoder_layers[-4]) / 4
        if not self.tunebert:
            # fix bert during training
            fl = fl.detach()

        # Visual-linguistic Fusion Module
        x = self.vlmodel(fv, fl)

        # Subject
        f_subject = self.f_subject(x)
        f_subject = f_subject.sigmoid()
        g_subject = self.g_subject(f_subject)
        g_subject = g_subject.sigmoid()
        m_s2o = self.m_s2o(x)
        m_s2o = m_s2o.sigmoid()

        # Object
        f_object = self.f_object(x)
        f_object = f_object.sigmoid()
        g_object = self.g_object(f_object)
        g_object = g_object.sigmoid()
        m_o2s = self.m_o2s(x)
        m_o2s = m_o2s.sigmoid()

        # final output for Subject
        output_subject = g_subject * f_subject + (torch.ones_like(g_subject) - g_subject) * m_o2s
        output_subject = self.output_subject(output_subject)
        output_subject = output_subject.sigmoid()

        # final output for Object
        output_object = g_object * f_object + (torch.ones_like(g_object) - g_object) * m_s2o
        output_object = self.output_object(output_object)
        output_object = output_object.sigmoid()

        return output_subject, output_object