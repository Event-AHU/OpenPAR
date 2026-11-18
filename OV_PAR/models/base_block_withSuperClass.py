import torch
import torch.nn as nn
import torch.nn.functional as F
from clip import clip
from models.prompt_learner import PromptLearner, TextEncoder, PromptLearner2
from models.crossattn import CrossAttentionLayer
from models.adapters import Adapter
from models.gate import ConditionalAttrGating
from models.ot_solver import Sinkhorn
from tools.sentences import build_training_group_sentences

def mmc_loss_rowwise(logits, mask, logits_mask=None):
    B, C = logits.shape
    if logits_mask is not None:
        logits = logits + (1 - logits_mask) * -1000000000.0
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    pos_count = torch.clamp(mask.sum(dim=1), min=1.0)
    loss = -(mask * log_prob).sum(dim=1) / pos_count
    return loss.mean()

@torch.no_grad()
def build_frozen_text_head(classnames, ntx, clip_model_teacher, device):
    prompt_prefix = ' '.join(['X'] * ntx)
    prompts = [prompt_prefix + ' ' + name + '.' for name in classnames]
    tokenized = clip.tokenize(prompts).to(device)
    embeds = clip_model_teacher.encode_text(tokenized)
    return F.normalize(embeds, dim=-1)

def prompt_anchor_loss(learned_txt: torch.Tensor, anchor_txt: torch.Tensor, margin: float=0.0) -> torch.Tensor:
    cos = (learned_txt * anchor_txt).sum(-1)
    return (1.0 - cos.clamp(min=margin)).mean()

class GroupPAR(nn.Module):

    def __init__(self, attributes, arg, dataset, dim=768, interaction_layer=[2, 5, 8, 11], num_layers=2, sample_weight=None):
        args = arg
        super().__init__()
        self.dataset = arg.dataset
        self.attributes = attributes
        self.combos_sentences = dataset.combos_sentences
        self.combos_labels = dataset.combos_labels
        self.group_sizes = dataset.group_sizes
        self.group_names = dataset.group_names
        self.group_descriptions = dataset.group_descriptions
        self.args = arg
        device = 'cuda'
        self.device = device
        self.topk = args.topk
        self.return_alllayers = True
        self.insert_layers = interaction_layer
        self.stage = args.stage
        self.inference_component_names = ['Match', 'CA', 'Global', 'OT-Loc']
        clip_model, _ = clip.load('CS-ViT-B/16', prompt_num=args.vp_len, device=device)
        clip_model_teacher, _ = clip.load('CS-ViT-B/16', prompt_num=0, device=device)
        self.clip_model = clip_model.float()
        self.prompt_learner = PromptLearner(args, attributes, clip_model_teacher, dataset, mode='attr-init')
        self.group_prompt_learner = PromptLearner(args, self.group_descriptions, clip_model_teacher, dataset, mode='group-init')
        self.group_sent_prompt_learner = PromptLearner2(args, clip_model_teacher)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts.cuda()
        self.group_tokenized_prompts = self.group_prompt_learner.tokenized_prompts.cuda()
        self.text_encoder = TextEncoder(clip_model_teacher).float()
        self.visual_adapters = nn.ModuleList([Adapter(self.clip_model.embed_dim, self.clip_model.embed_dim, dim=args.adapter_h) for _ in interaction_layer])
        self.crossattn = nn.ModuleList([CrossAttentionLayer(self.clip_model.embed_dim, dropout=0.1) for _ in interaction_layer])
        self.text_guide_gate = ConditionalAttrGating(self.clip_model.embed_dim, len(interaction_layer))
        self.text_guide_gate_group = ConditionalAttrGating(self.clip_model.embed_dim, len(interaction_layer))
        self.reg = 0.1
        self.temperature_loc = nn.Parameter(torch.tensor(0.002))
        self.temperature = nn.Parameter(1.0 / self.clip_model.logit_scale.exp())
        self.temperature_attr = nn.Parameter(1.0 / self.clip_model.logit_scale.exp())
        self.global_head = nn.Identity()
        self.clip_model_teacher = clip_model_teacher.float()
        self.group_embeddings_cache = {}
        self.group_labels_cache = {}
        self._precompute_text_embeddings()
        self.sample_weight = sample_weight
        with torch.no_grad():
            self.text_fea_teacher = build_frozen_text_head(self.attributes, args.context_len, self.clip_model_teacher, self.device)
            self.group_text_fea_teacher = build_frozen_text_head(self.group_descriptions, args.context_len, self.clip_model_teacher, self.device)

    @torch.no_grad()
    def _precompute_text_embeddings(self):
        self.group_embeddings_cache = {}
        print('Now, Pre-processing Group Sentence Embeddings!!!')
        for group_name, group_sentences in self.combos_sentences.items():
            flat_sentences = sum(group_sentences, [])
            if self.stage == 'stage1':
                tokenized = clip.tokenize(flat_sentences).to(self.device)
                with torch.no_grad():
                    sentence_embeds = self.clip_model.encode_text(tokenized)
            else:
                with torch.no_grad():
                    sentence_embeds = self.obtain_senc_emb(flat_sentences)
            self.group_embeddings_cache[group_name] = F.normalize(sentence_embeds, dim=-1)
            if group_name not in self.group_labels_cache:
                self.group_labels_cache[group_name] = self.combos_labels[group_name].to(self.device)
        print('Pre-Processing Group Sentence Embeddings Over!!!')

    def compute_infonce_loss(self, img_embeds, text_embeds, gt_label):
        batch_size = img_embeds.shape[0]
        logit_scale = self.clip_model.logit_scale.exp()
        logits_per_image = logit_scale * torch.matmul(img_embeds, text_embeds.t())
        logits_per_text = logit_scale * torch.matmul(text_embeds, img_embeds.t())
        labels = (gt_label.unsqueeze(1) == gt_label.unsqueeze(0)).all(dim=-1).float()
        loss_i = torch.nn.functional.cross_entropy(logits_per_image, labels)
        loss_t = torch.nn.functional.cross_entropy(logits_per_text, labels)
        loss = (loss_i + loss_t) / 2
        return loss / batch_size

    def build_weights(self, sim, dim=-1, temperature=0.1):
        with torch.no_grad():
            sim_max = sim.max(dim=dim)[0]
            weights = (sim_max / temperature).softmax(dim=-1)
        return weights

    def compute_ot_scores(self, img_loc, attr_embeds):
        attr_embeds = attr_embeds.unsqueeze(1)
        B, tok, dim = img_loc.shape
        C, gp, dim = attr_embeds.shape
        attr_embeds = attr_embeds.permute(1, 0, 2)
        img_loc = F.normalize(img_loc, dim=-1)
        attr_embeds = F.normalize(attr_embeds, dim=-1)
        sim = img_loc @ attr_embeds.transpose(1, 2)
        cost = (sim * self.clip_model.logit_scale.exp()).softmax(dim=-1)
        cost = 1.0 - cost
        reg = self.reg
        u = self.build_weights(sim.detach(), dim=2, temperature=0.1)
        v = torch.zeros((B, C), dtype=sim.dtype, device=sim.device).fill_(1.0 / C)
        with torch.no_grad():
            T = Sinkhorn(u, v, cost, reg=reg)
        if torch.isnan(T).any():
            raise ValueError('Found nan in OT matrix!')
        return T

    def obtain_txt_emb(self, mode='attribute'):
        if mode == 'attribute':
            prompt = self.prompt_learner()
            text_features = self.text_encoder(prompt, self.tokenized_prompts)
        else:
            prompt = self.group_prompt_learner()
            text_features = self.text_encoder(prompt, self.group_tokenized_prompts)
        return text_features

    def obtain_senc_emb(self, senc):
        prompt, tokenized_prompts = self.group_sent_prompt_learner(senc, self.clip_model)
        text_features = self.text_encoder(prompt, tokenized_prompts)
        return text_features

    def forward(self, imgs, imgname, base_indices, gt_labels, epoch):
        group_sentences = build_training_group_sentences(dataset_name=self.dataset, labels=gt_labels, base_group_sizes=self.group_sizes['base'], group_names=self.group_names)
        local_sentence_embeds = []
        for group_sentence in group_sentences:
            if self.stage == 'stage1':
                sentence_tokens = clip.tokenize(group_sentence).cuda()
                sentence_embeds = self.clip_model.encode_text(sentence_tokens.to(imgs.device))
            else:
                sentence_embeds = self.obtain_senc_emb(group_sentence)
            sentence_embeds = F.normalize(sentence_embeds, dim=-1)
            local_sentence_embeds.append(sentence_embeds)
        group_txt_feats = self.obtain_txt_emb(mode='group')
        base_group_embeds = group_txt_feats[:len(self.group_sizes['base'])]
        B = imgs.shape[0]
        attr_embeds = self.obtain_txt_emb()
        base_attr_embeds = attr_embeds[:len(base_indices)]
        x_vvv, x_ori, alllayers_ori = self.clip_model.visual(imgs.type(self.clip_model.dtype), return_alllayers=self.return_alllayers)
        img_glb_vvv, img_loc_vvv = (x_vvv[:, 0], x_vvv[:, 1:])
        img_glb_ori, img_loc_ori = (x_ori[:, 0], x_ori[:, 1:])
        attr_loc_embeds = []
        attr_loc_embeds_group = []
        T = self.compute_ot_scores(img_loc=img_loc_vvv, attr_embeds=base_attr_embeds)
        alllayers_ori_loc = [alllayers_ori[idx][:, 1:, :] for idx in self.insert_layers]
        for layer_idx in range(len(self.insert_layers)):
            visual_feat = self.visual_adapters[layer_idx](alllayers_ori_loc[layer_idx]) + alllayers_ori_loc[layer_idx]
            attr_loc_embed, attn_attr = self.crossattn[layer_idx](base_attr_embeds.unsqueeze(0).expand(B, -1, -1), visual_feat)
            attr_loc_embed_group, _ = self.crossattn[layer_idx](base_group_embeds.unsqueeze(0).expand(B, -1, -1), visual_feat)
            attr_loc_embeds.append(attr_loc_embed)
            attr_loc_embeds_group.append(attr_loc_embed_group)
        attr_loc_embed, _ = self.crossattn[-1](base_attr_embeds.unsqueeze(0).expand(B, -1, -1), img_loc_vvv)
        attr_loc_embed_group, _ = self.crossattn[-1](base_group_embeds.unsqueeze(0).expand(B, -1, -1), img_loc_vvv)
        fuse_feats = self.text_guide_gate(base_attr_embeds, attr_loc_embeds) + attr_loc_embed
        attr_loc_embeds.append(fuse_feats)
        fuse_feats_group = self.text_guide_gate_group(base_group_embeds, attr_loc_embeds_group) + attr_loc_embed_group
        attr_loc_embeds_group.append(fuse_feats_group)
        img_glb_ori = self.global_head(img_glb_ori + img_glb_vvv)
        img_glb_ori = F.normalize(img_glb_ori, dim=-1)
        fuse_feats = F.normalize(fuse_feats, dim=-1)
        base_attr_embeds = F.normalize(base_attr_embeds, dim=-1)
        loc_feat = F.normalize(img_loc_vvv, dim=-1)
        logits_ca = torch.diagonal(fuse_feats @ base_attr_embeds.t(), dim1=-1, dim2=-2)
        logits_glb = img_glb_ori @ base_attr_embeds.t()
        logits_locv = loc_feat @ base_attr_embeds.t()
        logits_locv = T * logits_locv
        logits_ca = logits_ca / self.temperature_attr
        logits_glb = logits_glb / self.temperature
        logits_locv = logits_locv.sum(dim=1) / self.temperature_loc
        contra_loss = 0.0
        for lidx, group_embs in enumerate(attr_loc_embeds_group):
            slice_start = 0
            for vis_feat, sentence_embeds, group_size in zip(group_embs.permute(1, 0, 2), local_sentence_embeds, self.group_sizes['base']):
                slice_end = slice_start + group_size
                vis_feat = F.normalize(vis_feat, dim=-1)
                contra_loss = contra_loss + self.compute_infonce_loss(vis_feat, sentence_embeds, gt_label=gt_labels[:, slice_start:slice_end])
                slice_start = slice_end
        contra_loss = contra_loss / (len(attr_loc_embeds_group) * len(local_sentence_embeds))
        cls_loss_glb = mmc_loss_rowwise(logits_glb, gt_labels)
        cls_loss_locv = mmc_loss_rowwise(logits_locv, gt_labels)
        score_ca = logits_ca.softmax(dim=-1)
        score_glb = logits_glb.softmax(dim=-1)
        score_locv = logits_locv.softmax(dim=-1)
        score = (score_ca + score_glb + score_locv) / 3
        cls_loss_ca = 0.0
        for sem_embs in attr_loc_embeds:
            sem_embs = F.normalize(sem_embs, dim=-1)
            logits = torch.diagonal(sem_embs @ base_attr_embeds.t(), dim1=-1, dim2=-2)
            logits = logits / self.temperature_attr
            cls_loss_ca = cls_loss_ca + mmc_loss_rowwise(logits, gt_labels)
        cls_loss_ca = cls_loss_ca / len(attr_loc_embeds)
        t_attr = self.text_fea_teacher[:len(base_indices)]
        L_anchor = prompt_anchor_loss(base_attr_embeds, t_attr.detach(), margin=0.1)
        return (score, [cls_loss_ca, cls_loss_glb, cls_loss_locv], contra_loss, L_anchor)

    @torch.no_grad()
    def match_result_group_topk(self, img_embeds, topk=5):
        group_pseudo_labels = []
        for gid, group_name in enumerate(self.group_names):
            group_sent_embeds = self.group_embeddings_cache[group_name]
            num_embeds = group_sent_embeds.shape[0]
            local_embeds_v = img_embeds[:, gid]
            logit_scale = self.clip_model.logit_scale.exp()
            logits = logit_scale * local_embeds_v @ group_sent_embeds.t()
            probs = torch.softmax(logits, dim=-1)
            k = min(topk, num_embeds)
            topk_probs, topk_indices = torch.topk(probs, k=k, dim=-1)
            selected_labels = self.group_labels_cache[group_name][topk_indices]
            labels_per_group = (selected_labels * topk_probs.unsqueeze(-1)).sum(dim=1)
            group_pseudo_labels.append(labels_per_group)
        pseudo_labels = torch.cat(group_pseudo_labels, dim=-1)
        return pseudo_labels

    def infer(self, epoch, imgs, imgname, base_indices, novel_indices):
        B = imgs.shape[0]
        x_vvv, x_ori, alllayers_ori = self.clip_model.visual(imgs.type(self.clip_model.dtype), return_alllayers=self.return_alllayers)
        img_glb_vvv, img_loc_vvv = (x_vvv[:, 0], x_vvv[:, 1:])
        img_glb_ori, img_loc_ori = (x_ori[:, 0], x_ori[:, 1:])
        attr_loc_embeds_all = []
        attr_loc_embeds_group = []
        attr_loc_embeds_novel = []
        attr_embeds = self.obtain_txt_emb()
        group_embeds = self.obtain_txt_emb(mode='group')
        novel_attr_embeds = attr_embeds[len(base_indices):]
        T = self.compute_ot_scores(img_loc=img_loc_vvv, attr_embeds=attr_embeds)
        T_novel = self.compute_ot_scores(img_loc=img_loc_vvv, attr_embeds=novel_attr_embeds)
        alllayers_ori_loc = [alllayers_ori[idx][:, 1:, :] for idx in self.insert_layers]
        for layer_idx in range(len(self.insert_layers)):
            visual_feat = self.visual_adapters[layer_idx](alllayers_ori_loc[layer_idx]) + alllayers_ori_loc[layer_idx]
            attr_loc_embed, attn_attr = self.crossattn[layer_idx](attr_embeds.unsqueeze(0).expand(B, -1, -1), visual_feat)
            attr_loc_embed_novel, _ = self.crossattn[layer_idx](novel_attr_embeds.unsqueeze(0).expand(B, -1, -1), visual_feat)
            attr_loc_embeds_all.append(attr_loc_embed)
            attr_loc_embeds_novel.append(attr_loc_embed_novel)
            attr_loc_embed_group, _ = self.crossattn[layer_idx](group_embeds.unsqueeze(0).expand(B, -1, -1), visual_feat)
            attr_loc_embeds_group.append(attr_loc_embed_group)
        attr_loc_embeds_all = self.text_guide_gate(attr_embeds, attr_loc_embeds_all)
        attr_loc_embeds_novel = self.text_guide_gate(novel_attr_embeds, attr_loc_embeds_novel)
        attr_loc_embeds_group = self.text_guide_gate_group(group_embeds, attr_loc_embeds_group)
        img_glb_ori = self.global_head(img_glb_ori + img_glb_vvv)
        img_glb_ori = F.normalize(img_glb_ori, dim=-1)
        attr_loc_embeds_all = F.normalize(attr_loc_embeds_all, dim=-1)
        attr_loc_embeds_novel = F.normalize(attr_loc_embeds_novel, dim=-1)
        attr_loc_embeds_group = F.normalize(attr_loc_embeds_group, dim=-1)
        attr_embeds = F.normalize(attr_embeds, dim=-1)
        novel_attr_embeds = F.normalize(novel_attr_embeds, dim=-1)
        loc_feat = F.normalize(img_loc_vvv, dim=-1)
        match_logits = self.match_result_group_topk(attr_loc_embeds_group, self.topk)
        match_scores = torch.cat([match_logits[:, base_indices], match_logits[:, novel_indices]], dim=-1).softmax(dim=-1)
        match_scores_novel = match_logits[:, novel_indices].softmax(dim=-1)
        logits1 = torch.diagonal(attr_loc_embeds_all @ attr_embeds.t(), dim1=-1, dim2=-2)
        logits2 = img_glb_ori @ attr_embeds.t()
        logits3 = loc_feat @ attr_embeds.t()
        logits3 = T * logits3
        logits1 = logits1 / self.temperature_attr
        logits2 = logits2 / self.temperature
        logits3 = logits3.sum(dim=1) / self.temperature_loc
        logits1 = logits1.softmax(dim=-1)
        logits2 = logits2.softmax(dim=-1)
        logits3 = logits3.softmax(dim=-1)
        logits = (match_scores + sum([logits1, logits2, logits3])) / 4
        novel_logits1 = torch.diagonal(attr_loc_embeds_novel @ novel_attr_embeds.t(), dim1=-1, dim2=-2)
        novel_logits2 = img_glb_ori @ novel_attr_embeds.t()
        novel_logits3 = loc_feat @ novel_attr_embeds.t()
        novel_logits3 = T_novel * novel_logits3
        novel_logits1 = novel_logits1 / self.temperature_attr
        novel_logits2 = novel_logits2 / self.temperature
        novel_logits3 = novel_logits3.sum(dim=1) / self.temperature_loc
        novel_logits1 = novel_logits1.softmax(dim=-1)
        novel_logits2 = novel_logits2.softmax(dim=-1)
        novel_logits3 = novel_logits3.softmax(dim=-1)
        novel_logits = (match_scores_novel + sum([novel_logits1, novel_logits2, novel_logits3])) / 4
        primary_component = match_scores
        primary_novel_component = match_scores_novel
        return (logits, novel_logits, [primary_component, logits1, logits2, logits3], [primary_novel_component, novel_logits1, novel_logits2, novel_logits3])
