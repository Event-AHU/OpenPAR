import torch
import torch.nn as nn
import torch.nn.functional as F
from clip import clip
from models.vit import *
from config import argument_parser
from .UniGNN import UniGNN
from torch_geometric.utils import dense_to_sparse
import json

parser = argument_parser()
args = parser.parse_args()

class TransformerClassifier(nn.Module):
    def __init__(self, clip_model, attr_num, attributes, dim=768,
                 pretrain_path='/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb6/DATA/wushujuan/PromptPAR/jx_vit_base_p16_224-80ecf9dd.pth'):
        super().__init__()
        self.attr_num = attr_num
        self.dim = dim
        
        self.region_text_features = None
        self.text_ranges = None
        self.register_buffer('global_hg', None)
        
        self.word_embed = nn.Linear(clip_model.visual.output_dim, dim)
        self.visual_embed = nn.Linear(clip_model.visual.output_dim, dim)

        vit = vit_base()
        vit.load_param(pretrain_path)
        self.norm = vit.norm
        self.weight_layer = nn.ModuleList([nn.Linear(dim, 1) for _ in range(self.attr_num)])
        
        json_path = '/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb6/DATA/wushujuan/PromptPAR/dataset/hy/rapv1.json'
        with open(json_path, 'r', encoding='utf-8') as f:
            attributes_dict = json.load(f)
        self.region_texts = {
            'body': attributes_dict['body'],
            'head': attributes_dict['head'],
            'upper': attributes_dict['upper'],
            'lower': attributes_dict['lower'],
            'foot': attributes_dict['foot']
        }

        self.region_tokens = {}
        for region, texts in self.region_texts.items():
            self.region_tokens[region] = clip.tokenize(texts).to("cuda")

        self.bn = nn.BatchNorm1d(self.attr_num)

        fusion_len = self.attr_num + 257 + args.vis_prompt
        if not args.use_mm_former:
            print('Without MM-former, Using MLP Instead')
            self.linear_layer = nn.Linear(fusion_len, self.attr_num)
        else:
            self.blocks = vit.blocks[-args.mm_layers:]

        class UniGNNArgs:
            model_name = 'UniSAGE'
            first_aggregate = 'mean'
            second_aggregate = 'sum'
            add_self_loop = False
            use_norm = True
            activation = 'relu'
            nlayer = 2
            nhid = 768
            nhead = 1
            dropout = 0.6
            input_drop = 0.6
            attn_drop = 0.6
            degE = None  
            degV = None
            
        gnn_args = UniGNNArgs()
        self.hgraph_shared = UniGNN(
            args=gnn_args,
            nfeat=dim,
            nhid=gnn_args.nhid,
            nclass=dim,
            nlayer=gnn_args.nlayer,
            nhead=gnn_args.nhead,
            V=None,
            E=None
        )

        self.data_loaded = False
        self.gnn_args = gnn_args

    def _load_global_data(self, device):
        if not self.data_loaded:
            data_path = '/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb6/DATA/wushujuan/PromptPAR/dataset/hy/rapv1_unified_clip_features_8.pt'
            data = torch.load(data_path, weights_only=False)

            features = data['features'].to(device)
            labels = data['labels']

            edge_index, _ = dense_to_sparse(labels.float())
            V = edge_index[0].to(device)
            E = edge_index[1].to(device)

            num_vertices = features.shape[0]
            num_edges = E.max().item() + 1
            
            edge_degrees = torch.zeros(num_edges, device=device)
            for e in range(num_edges):
                edge_mask = (E == e)
                edge_degrees[e] = edge_mask.sum().float()
            
            degE = torch.pow(edge_degrees, -0.5)
            degE[torch.isinf(degE)] = 0.
            degE = degE.unsqueeze(-1) 

            vertex_degrees = torch.zeros(num_vertices, device=device)
            for v in range(num_vertices):
                vertex_mask = (V == v)
                vertex_degrees[v] = vertex_mask.sum().float()

            degV = torch.pow(vertex_degrees, -0.5)
            degV[torch.isinf(degV)] = 0.
            degV = degV.unsqueeze(-1)  # [num_vertices, 1]
            

            self.gnn_args.degE = degE
            self.gnn_args.degV = degV

            self.hgraph_global = UniGNN(
                args=self.gnn_args,
                nfeat=features.shape[1],
                nhid=self.gnn_args.nhid,
                nclass=self.dim,
                nlayer=self.gnn_args.nlayer,
                nhead=self.gnn_args.nhead,
                V=V,
                E=E
            ).to(device)
            
            with torch.no_grad():
                self.global_hg = self.hgraph_global(features).detach()
            
            self.data_loaded = True

    def _precompute_text_features(self, clip_model):
        if self.region_text_features is None:
            all_tokens = []
            region_slices = {}
            start = 0
            for region, tokens in self.region_tokens.items():
                all_tokens.append(tokens)
                end = start + tokens.shape[0]
                region_slices[region] = (start, end)
                start = end

            all_tokens = torch.cat(all_tokens, dim=0)
            with torch.no_grad():
                all_text_features = clip_model.encode_text(all_tokens).float()

            region_text_features = {}
            for region, (s, e) in region_slices.items():
                region_text_features[region] = all_text_features[s:e]
            
            self.region_text_features = region_text_features

         
    def build_unified_hypergraph(self, global_patches, part_indices_info, region_texts_dict, threshold=0.6):
        all_nodes = []
        all_nodes.append(global_patches)

        text_start_idx = global_patches.shape[0]  
        text_count = 0
        
        region_names = ['body', 'head', 'upper', 'lower', 'foot']
        for region_name in region_names:
            texts = region_texts_dict[region_name]
            all_nodes.append(texts)
            text_count += texts.shape[0]
        
        all_nodes = torch.cat(all_nodes, dim=0)

        V_list = []
        E_list = []
        hyperedge_counter = 0

        region_mapping = {0: 'body', 1: 'head', 2: 'upper', 3: 'lower', 4: 'foot'}

        for info in part_indices_info:
            if info.get('part_idx', -1) >= 0: 
                region_idx = info['part_idx']
                region_name = region_mapping.get(region_idx)

                original_start = info.get('original_start', 0)
                original_end = info.get('original_end', 0)
                region_patch_indices = list(range(original_start, original_end))

                region_texts = region_texts_dict[region_name]
                region_patches = global_patches[region_patch_indices]

                if region_patches.dtype != region_texts.dtype:
                    region_patches = region_patches.to(region_texts.dtype)
                
                sim_matrix = torch.matmul(region_patches, region_texts.T)
                for text_idx in range(region_texts.shape[0]):
                    similarity_values = sim_matrix[:, text_idx]
                    valid_patch_indices = torch.where(similarity_values > threshold)[0]
         
                    if valid_patch_indices.numel() > 0:
                        for patch_idx in valid_patch_indices:
                            global_patch_idx = region_patch_indices[patch_idx]
                            V_list.append(global_patch_idx)
                            E_list.append(hyperedge_counter)

                        text_count_before = 0
                        for i, name in enumerate(region_names):
                            if name == region_name:
                                break
                            text_count_before += region_texts_dict[name].shape[0]
                        
                        text_global_idx = text_start_idx + text_count_before + text_idx
                        V_list.append(text_global_idx)
                        E_list.append(hyperedge_counter)
                        
                        hyperedge_counter += 1
                   
        if len(V_list) == 0:
            return all_nodes, None, None
        
        V = torch.tensor(V_list, device=all_nodes.device, dtype=torch.long)
        E = torch.tensor(E_list, device=all_nodes.device, dtype=torch.long)
        
        return all_nodes, V, E
    
    def forward(self, imgs, clip_model):
        device = imgs.device
        b_s = imgs.shape[0]

        self._load_global_data(device)
        self._precompute_text_features(clip_model)

        clip_image_features, all_class, attnmap, part_patch_tokens, part_indices_info = clip_model.visual(imgs.type(clip_model.dtype))
    
        enhanced_features = None
        final_similarity = None
        region_names = ['body', 'head', 'upper', 'lower', 'foot']

        if args.use_div and part_patch_tokens is not None:
            all_region_texts = torch.cat([self.region_text_features[region] for region in region_names], dim=0)
            final_similarity, logits_per_image = clip_model.forward_aggregate(all_class, all_region_texts)
            
            enhanced_batch = []
                
            for b in range(b_s):
                global_patches = part_patch_tokens[0][b]  # [256, dim]
                
                all_nodes, V, E = self.build_unified_hypergraph(
                    global_patches, part_indices_info, self.region_text_features, threshold=0.6
                )
                if V is not None and E is not None:
                    num_vertices = all_nodes.shape[0]
                    num_edges = E.max().item() + 1

                    edge_degrees = torch.zeros(num_edges, device=all_nodes.device)
                    for e in range(num_edges):
                        edge_mask = (E == e)
                        edge_degrees[e] = edge_mask.sum().float()
                    
                    degE = torch.pow(edge_degrees, -0.5)
                    degE[torch.isinf(degE)] = 0.
                    degE = degE.unsqueeze(-1)

                    vertex_degrees = torch.zeros(num_vertices, device=all_nodes.device)
                    for v in range(num_vertices):
                        vertex_mask = (V == v)
                        vertex_degrees[v] = vertex_mask.sum().float()
                    
                    degV = torch.pow(vertex_degrees, -0.5)
                    degV[torch.isinf(degV)] = 0.
                    degV = degV.unsqueeze(-1)

                    self.hgraph_shared.V = V
                    self.hgraph_shared.E = E
                    self.hgraph_shared.args.degE = degE
                    self.hgraph_shared.args.degV = degV
                    
                    enhanced_nodes = self.hgraph_shared(all_nodes.float())
                    enhanced_batch.append(enhanced_nodes.unsqueeze(0))
                else:
                    enhanced_batch.append(all_nodes.unsqueeze(0))
            
            if enhanced_batch:
                node_counts = [feat.shape[1] for feat in enhanced_batch]
                min_nodes = int(torch.min(torch.tensor(node_counts)).item())  
                
                padded_batch = []
                for feat in enhanced_batch:
                    if feat.shape[1] > min_nodes:
                        indices = torch.randperm(feat.shape[1], device=feat.device)[:min_nodes]
                        feat = feat[:, indices, :]
                    padded_batch.append(feat)
                enhanced_features = torch.cat(padded_batch, dim=0)

        cls_token = all_class[:, 0].float()
        global_hg = self.global_hg.float()

        global_hg_norm = F.normalize(global_hg, dim=-1)
        clip_img_norm = F.normalize(cls_token, dim=-1)  
        sim_matrix = torch.matmul(clip_img_norm, global_hg_norm.T) 

        with torch.no_grad():
            topk_values, topk_indices = torch.topk(sim_matrix, k=10, dim=1)  # [B, K]

        topk_global_features = global_hg[topk_indices]  
 
        x = torch.cat([
            enhanced_features,
            topk_global_features,
            self.visual_embed(clip_image_features.float())
        ], dim=1)
        
        if args.use_mm_former:
            for blk in self.blocks:
                x = blk(x)
        else:
            x = x.permute(0, 2, 1)
            x = self.linear_layer(x)
            x = x.permute(0, 2, 1)

        x = self.norm(x)
        logits = torch.cat([self.weight_layer[i](x[:, i, :]) for i in range(self.attr_num)], dim=1)
        bn_logits = self.bn(logits)

        return bn_logits, final_similarity
            # import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from clip import clip
# from models.vit import *
# from config import argument_parser
# from .UniGNN import UniGNN,UniGNN
# from torch_geometric.utils import dense_to_sparse
# import json

# parser = argument_parser()
# args = parser.parse_args()

# class TransformerClassifier(nn.Module):
#     def __init__(self, clip_model, attr_num, attributes, dim=768,
#                  pretrain_path='/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb6/DATA/wushujuan/PromptPAR/jx_vit_base_p16_224-80ecf9dd.pth'):
#         super().__init__()
#         self.attr_num = attr_num
#         self.dim = dim
        
#         self.region_text_features = None
#         self.text_ranges = None
#         self.register_buffer('global_hg', None)
        
#         self.word_embed = nn.Linear(clip_model.visual.output_dim, dim)

#         self.visual_embed = nn.Linear(clip_model.visual.output_dim, dim)


#         vit = vit_base()
#         vit.load_param(pretrain_path)
#         self.norm = vit.norm
#         self.weight_layer = nn.ModuleList([nn.Linear(dim, 1) for _ in range(self.attr_num)])
        
#         json_path = '/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb6/DATA/wushujuan/PromptPAR/dataset/hy/rapv1.json'
#         with open(json_path, 'r', encoding='utf-8') as f:
#             attributes_dict = json.load(f)
#         self.region_texts = {
#             'body': attributes_dict['body'],
#             'head': attributes_dict['head'],
#             'upper': attributes_dict['upper'],
#             'lower': attributes_dict['lower'],
#             'foot': attributes_dict['foot']
#         }

#         self.region_tokens = {}
#         for region, texts in self.region_texts.items():
#             self.region_tokens[region] = clip.tokenize(texts).to("cuda")

#         self.bn = nn.BatchNorm1d(self.attr_num)

#         fusion_len = self.attr_num + 257 + args.vis_prompt
#         if not args.use_mm_former:
#             print('Without MM-former, Using MLP Instead')
#             self.linear_layer = nn.Linear(fusion_len, self.attr_num)
#         else:
#             self.blocks = vit.blocks[-args.mm_layers:]

#         class UniGNNArgs:
#             model_name = 'UniGAT'
#             first_aggregate = 'mean'
#             second_aggregate = 'sum'
#             add_self_loop = False
#             use_norm = True
#             activation = 'relu'
#             nlayer = 2
#             nhid = self.dim
#             nhead = 1
#             dropout = 0.6
#             input_drop = 0.6
#             attn_drop = 0.6
       
#         gnn_args = UniGNNArgs()
#         self.hgraph_shared = UniGNN(
#             args=gnn_args,
#             nfeat=dim,
#             nhid=gnn_args.nhid,
#             nclass=dim,
#             nlayer=gnn_args.nlayer,
#             nhead=gnn_args.nhead,
#             V=None,
#             E=None
#         )

#         self.data_loaded = False
#         self.gnn_args = gnn_args

#     def _load_global_data(self, device):
#         if not self.data_loaded:
#             data_path = '/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb6/DATA/wushujuan/PromptPAR/dataset/hy/rapv1_unified_clip_features_4.pt'
#             data = torch.load(data_path, weights_only=False)

#             features = data['features'].to(device)
#             labels = data['labels']

#             edge_index, _ = dense_to_sparse(labels.float())
#             V = edge_index[0].to(device)
#             E = edge_index[1].to(device)

#             self.hgraph_global = UniGNN(
#                 args=self.gnn_args,
#                 nfeat=features.shape[1],
#                 nhid=self.gnn_args.nhid,
#                 nclass=self.dim,
#                 nlayer=self.gnn_args.nlayer,
#                 nhead=self.gnn_args.nhead,
#                 V=V,
#                 E=E
#             ).to(device)
            
#             with torch.no_grad():
#                 self.global_hg = self.hgraph_global(features).detach()
            
#             self.data_loaded = True

#     def _precompute_text_features(self, clip_model):
#         if self.region_text_features is None:
#             all_tokens = []
#             region_slices = {}
#             start = 0
#             for region, tokens in self.region_tokens.items():
#                 all_tokens.append(tokens)
#                 end = start + tokens.shape[0]
#                 region_slices[region] = (start, end)
#                 start = end

#             all_tokens = torch.cat(all_tokens, dim=0)
#             with torch.no_grad():
#                 all_text_features = clip_model.encode_text(all_tokens).float()

#             region_text_features = {}
#             for region, (s, e) in region_slices.items():
#                 region_text_features[region] = all_text_features[s:e]
            
#             self.region_text_features = region_text_features

         
#     def build_unified_hypergraph(self, global_patches, part_indices_info, region_texts_dict, threshold=0.6):
#         all_nodes = []
#         all_nodes.append(global_patches)

#         text_start_idx = global_patches.shape[0]  
#         text_count = 0
        
#         region_names = ['body', 'head', 'upper', 'lower', 'foot']
#         for region_name in region_names:
#             texts = region_texts_dict[region_name]
#             all_nodes.append(texts)
#             text_count += texts.shape[0]
        
#         all_nodes = torch.cat(all_nodes, dim=0)

#         V_list = []
#         E_list = []
#         hyperedge_counter = 0

#         region_mapping = {0: 'body', 1: 'head', 2: 'upper', 3: 'lower', 4: 'foot'}

#         for info in part_indices_info:
#             if info.get('part_idx', -1) >= 0: 
#                 region_idx = info['part_idx']
#                 region_name = region_mapping.get(region_idx)

#                 original_start = info.get('original_start', 0)
#                 original_end = info.get('original_end', 0)
#                 region_patch_indices = list(range(original_start, original_end))

#                 region_texts = region_texts_dict[region_name]
#                 region_patches = global_patches[region_patch_indices]

#                 if region_patches.dtype != region_texts.dtype:
#                     region_patches = region_patches.to(region_texts.dtype)
                
#                 sim_matrix = torch.matmul(region_patches, region_texts.T)
#                 for text_idx in range(region_texts.shape[0]):
#                     similarity_values = sim_matrix[:, text_idx]
#                     valid_patch_indices = torch.where(similarity_values > threshold)[0]
         
#                     if valid_patch_indices.numel() > 0:
#                         for patch_idx in valid_patch_indices:
#                             global_patch_idx = region_patch_indices[patch_idx]
#                             V_list.append(global_patch_idx)
#                             E_list.append(hyperedge_counter)

#                         text_count_before = 0
#                         for i, name in enumerate(region_names):
#                             if name == region_name:
#                                 break
#                             text_count_before += region_texts_dict[name].shape[0]
                        
#                         text_global_idx = text_start_idx + text_count_before + text_idx
#                         V_list.append(text_global_idx)
#                         E_list.append(hyperedge_counter)
                        
#                         hyperedge_counter += 1
                   
#         if len(V_list) == 0:
#             return all_nodes, None, None
        
#         V = torch.tensor(V_list, device=all_nodes.device, dtype=torch.long)
#         E = torch.tensor(E_list, device=all_nodes.device, dtype=torch.long)
        
#         return all_nodes, V, E
    
#     def forward(self, imgs, clip_model):
#         device = imgs.device
#         b_s = imgs.shape[0]

#         self._load_global_data(device)
#         self._precompute_text_features(clip_model)

#         clip_image_features, all_class, attnmap, part_patch_tokens, part_indices_info = clip_model.visual(imgs.type(clip_model.dtype))
    
#         enhanced_features = None
#         final_similarity = None
#         region_names = ['body', 'head', 'upper', 'lower', 'foot']

#         if args.use_div and part_patch_tokens is not None:
#             all_region_texts = torch.cat([self.region_text_features[region] for region in region_names], dim=0)
#             final_similarity, logits_per_image = clip_model.forward_aggregate(all_class, all_region_texts)
  
#             #text_ranges = self.text_ranges

#             # filtered_region_texts = {}
#             # for i, region in enumerate(region_names):
#             #     region_sim = logits_per_image[:, i, :]
#             #     start, end = text_ranges[region]
#             #     region_text_sim = region_sim[:, start:end]
                
#             #     filtered_texts_list = []
#             #     for b in range(b_s):
#             #         sim_values = region_text_sim[b]
#             #         num_texts = len(sim_values)
#             #         num_to_keep = max(1, int(num_texts * 0.6))
                    
#             #         _, topk_indices = torch.topk(sim_values, k=num_to_keep)
#             #         topk_texts = self.region_text_features[region][topk_indices]
#             #         filtered_texts_list.append(topk_texts)
                
#             #     filtered_region_texts[region] = filtered_texts_list
            
#             enhanced_batch = []
            
            
                
#             for b in range(b_s):
#                 global_patches = part_patch_tokens[0][b]  # [256, dim]
                
#                 all_nodes, V, E = self.build_unified_hypergraph(
#                     global_patches, part_indices_info, self.region_text_features, threshold=0.6
#                 )
#                 if V is not None and E is not None:
#                     self.hgraph_shared.V = V
#                     self.hgraph_shared.E = E
#                     enhanced_nodes = self.hgraph_shared(all_nodes.float())
#                     enhanced_batch.append(enhanced_nodes.unsqueeze(0))
#                 else:
#                     enhanced_batch.append(all_nodes.unsqueeze(0))
            
#             if enhanced_batch:
#                 node_counts = [feat.shape[1] for feat in enhanced_batch]
#                 min_nodes = int(torch.min(torch.tensor(node_counts)).item())  
                
#                 padded_batch = []
#                 for feat in enhanced_batch:
#                     if feat.shape[1] > min_nodes:
#                         indices = torch.randperm(feat.shape[1], device=feat.device)[:min_nodes]
#                         feat = feat[:, indices, :]
#                     padded_batch.append(feat)
#                 enhanced_features = torch.cat(padded_batch, dim=0)



#         cls_token = all_class[:, 0].float()
#         global_hg = self.global_hg.float()

#         global_hg_norm = F.normalize(global_hg, dim=-1)
#         clip_img_norm = F.normalize(cls_token, dim=-1)  
#         sim_matrix = torch.matmul(clip_img_norm, global_hg_norm.T) 

#         with torch.no_grad():
#             topk_values, topk_indices = torch.topk(sim_matrix, k=10, dim=1)  # [B, K]

#         topk_global_features = global_hg[topk_indices]  
 

#         x = torch.cat([
#             enhanced_features,
#             topk_global_features,
#             self.visual_embed(clip_image_features.float())
#         ], dim=1)
        
#         if args.use_mm_former:
#             for blk in self.blocks:
#                 x = blk(x)
#         else:
#             x = x.permute(0, 2, 1)
#             x = self.linear_layer(x)
#             x = x.permute(0, 2, 1)

#         x = self.norm(x)
#         logits = torch.cat([self.weight_layer[i](x[:, i, :]) for i in range(self.attr_num)], dim=1)
#         bn_logits = self.bn(logits)

#         return bn_logits, final_similarity

