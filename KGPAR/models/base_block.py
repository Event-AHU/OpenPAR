# import torch.nn as nn
# import torch
# from clip import clip
# from models.vit import *
# from config import argument_parser
# parser = argument_parser()
# args = parser.parse_args()
# class TransformerClassifier(nn.Module):
#     def __init__(self, clip_model, attr_num, attributes, dim=768, pretrain_path='/rydata/wushujuan/VTB/jx_vit_base_p16_224-80ecf9dd.pth'):
#         super().__init__()
#         super().__init__()
#         self.attr_num = attr_num
#         self.word_embed = nn.Linear(clip_model.visual.output_dim, dim)
#         self.visual_embed = nn.Linear(clip_model.visual.output_dim, dim)
#         vit = vit_base()
#         vit.load_param(pretrain_path)
#         self.norm = vit.norm
#         self.weight_layer = nn.ModuleList([nn.Linear(dim, 1) for i in range(self.attr_num)])
#         self.dim = dim
#         self.text = clip.tokenize(attributes).to("cuda")
#         self.bn = nn.BatchNorm1d(self.attr_num)
#         fusion_len = self.attr_num + 257 + args.vis_prompt
#         if not args.use_mm_former :
#             print('Without MM-former, Using MLP Instead')
#             self.linear_layer = nn.Linear(fusion_len, self.attr_num)
#         else:
#             self.blocks = vit.blocks[-args.mm_layers:]
#     def forward(self,imgs,clip_model):
#         b_s=imgs.shape[0]
#         clip_image_features,all_class,attenmap,part_patch_tokens=clip_model.visual(imgs.type(clip_model.dtype))
#         text_features = clip_model.encode_text(self.text).to("cuda").float()
#         if args.use_div:
#             final_similarity,logits_per_image = clip_model.forward_aggregate(all_class,text_features)
#         else : 
#             final_similarity = None
#         textual_features = self.word_embed(text_features).expand(b_s, self.attr_num, self.dim)
#         x = torch.cat([textual_features, self.visual_embed(clip_image_features.float())], dim=1)
        
#         if args.use_mm_former:
#             for blk in self.blocks:
#                 x = blk(x)
#         else :# using linear layer fusion
#             x = x.permute(0, 2, 1)
#             x= self.linear_layer(x)
#             x = x.permute(0, 2, 1)
            
#         x = self.norm(x)
#         logits = torch.cat([self.weight_layer[i](x[:, i, :]) for i in range(self.attr_num)], dim=1)
#         bn_logits = self.bn(logits)
        
        
#         return bn_logits,final_similarity
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from clip import clip
# from models.vit import *
# from config import argument_parser
# from .UniGNN import UniGNN
# from torch_geometric.utils import dense_to_sparse
# parser = argument_parser()
# args = parser.parse_args()

# class TransformerClassifier(nn.Module):
#     def __init__(self, clip_model, attr_num, attributes, dim=768,
#                  pretrain_path='/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb6/DATA/wushujuan/PromptPAR/jx_vit_base_p16_224-80ecf9dd.pth'):
#         super().__init__()
#         self.attr_num = attr_num
#         self.dim = dim
#         self.word_embed = nn.Linear(clip_model.visual.output_dim, dim)
#         self.visual_embed = nn.Linear(clip_model.visual.output_dim, dim)
#         vit = vit_base()
#         vit.load_param(pretrain_path)
#         self.norm = vit.norm
#         self.weight_layer = nn.ModuleList([nn.Linear(dim, 1) for _ in range(self.attr_num)])
#         self.text = clip.tokenize(attributes).to("cuda")
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
#             nhid = 1024
#             nhead = 4
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
       
#         data_path = '/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb6/DATA/wushujuan/PromptPAR/dataset/rapv1_unified_clip_features.pt'
#         data = torch.load(data_path)

#         features = data['features']      
#         labels = data['labels']            
#         V, E = dense_to_sparse(labels)  
#         self.hgraph_global = UniGNN(
#             args=gnn_args,
#             nfeat=1024,
#             nhid=gnn_args.nhid,
#             nclass=dim,
#             nlayer=gnn_args.nlayer,
#             nhead=gnn_args.nhead,
#             V=None,
#             E=None
#         )
#         self.global_hg= self.hgraph_global(features)
#     def extract_region_texts_from_logits(self, logits_per_image, text_features, threshold=0.5):
       
#         batch_size, total_regions, attr_num = logits_per_image.shape
#         used_text_indices = [set() for _ in range(batch_size)] 
#         region_text_indices_filtered = []

#         for region_idx in range(total_regions):
#             region_indices_batch = []
#             region_logits = logits_per_image[:, region_idx, :]
#             for batch_idx in range(batch_size):
#                 sample_logits = region_logits[batch_idx]
#                 high_sim_indices = torch.where(sample_logits > threshold)[0].tolist()
#                 new_indices = [idx for idx in high_sim_indices if idx not in used_text_indices[batch_idx]]
#                 used_text_indices[batch_idx].update(new_indices)
#                 region_indices_batch.append(torch.tensor(new_indices, device=text_features.device, dtype=torch.long))
#             region_text_indices_filtered.append(region_indices_batch)
#         return region_text_indices_filtered

#     def build_hypergraph_from_regions(self, text_nodes_b, all_patches_b, filtered_region_indices, region_patches_list, threshold=0.6):
   
#         if len(text_nodes_b) == 0 or len(region_patches_list) == 0:
#             return None, None, None
   
#         if all_patches_b.dim() == 3:
#             all_patches_b = all_patches_b.view(-1, all_patches_b.shape[-1])
#         if text_nodes_b.dim() == 3:
#             text_nodes_b = text_nodes_b.view(-1, text_nodes_b.shape[-1])
        
#         nodes = torch.cat([text_nodes_b, all_patches_b], dim=0)
#         num_text_nodes = text_nodes_b.shape[0]
        
#         V_list, E_list = [], []
#         hyperedge_idx = 0
#         text_idx_offset = 0

#         for region_idx, region_patches in enumerate(region_patches_list):
#             region_text_indices = filtered_region_indices[region_idx]
            
#             if len(region_text_indices) == 0 or region_patches.shape[0] == 0:
#                 continue

#             if region_patches.dim() == 3:
#                 region_patches = region_patches.view(-1, region_patches.shape[-1])

#             region_text_features = text_nodes_b[text_idx_offset:text_idx_offset+len(region_text_indices)]
#             text_idx_offset += len(region_text_indices)
            
#             text_norm = F.normalize(region_text_features.float(), dim=-1)
#             patch_norm = F.normalize(region_patches.float(), dim=-1)
#             similarity_matrix = torch.mm(text_norm, patch_norm.t())
            
#             for text_local_idx, text_global_idx in enumerate(region_text_indices):
#                 similarities = similarity_matrix[text_local_idx] 
#                 high_sim_patches = torch.where(similarities > threshold)[0]
                
#                 if len(high_sim_patches) > 0:
#                     text_node_idx = text_global_idx.item() if hasattr(text_global_idx, 'item') else text_global_idx
#                     V_list.append(text_node_idx)
#                     E_list.append(hyperedge_idx)

#                     region_patch_start = sum([rp.shape[0] if rp.dim() == 2 else rp.view(-1, rp.shape[-1]).shape[0] 
#                                             for rp in region_patches_list[:region_idx]])
#                     for patch_local_idx in high_sim_patches:
#                         patch_global_idx = num_text_nodes + region_patch_start + patch_local_idx.item()
#                         V_list.append(patch_global_idx)
#                         E_list.append(hyperedge_idx)
                    
#                     hyperedge_idx += 1
        
#         if len(V_list) == 0:
#             return nodes, None, None
            
#         V = torch.tensor(V_list, dtype=torch.long, device=nodes.device)
#         E = torch.tensor(E_list, dtype=torch.long, device=nodes.device)
        
#         return nodes, V, E

#     def forward(self, imgs, clip_model):
#         b_s = imgs.shape[0]
#         clip_image_features, all_class, attenmap, part_patch_tokens = clip_model.visual(imgs.type(clip_model.dtype))
#         text_features = clip_model.encode_text(self.text).to("cuda").float()

#         enhanced_features = None
#         final_similarity = None

#         if args.use_div:
#             final_similarity, logits_per_image = clip_model.forward_aggregate(all_class, text_features)
#             filtered_region_indices = self.extract_region_texts_from_logits(logits_per_image, text_features, threshold=0.6)

#             if part_patch_tokens is not None:
#                 enhanced_batch = []
#                 for b in range(b_s):
#                     text_nodes_list = []
#                     current_batch_region_indices = []
#                     for region_idx in range(len(filtered_region_indices)):
#                         region_indices = filtered_region_indices[region_idx][b]
#                         current_batch_region_indices.append(region_indices)
#                         if len(region_indices) > 0:
#                             text_nodes_list.append(text_features[region_indices])
                    
#                     if len(text_nodes_list) > 0:
#                         text_nodes_b = torch.cat(text_nodes_list, dim=0)
#                         region_patches_list = []
#                         for region_tokens in part_patch_tokens:
#                             if region_tokens.dim() == 3 and region_tokens.shape[0] == b_s:
#                                 region_patches_list.append(region_tokens[b])
#                             elif region_tokens.dim() == 2:
#                                 region_patches_list.append(region_tokens)
#                             else:
#                                 region_patches_list.append(region_tokens[b] if region_tokens.shape[0] > b else region_tokens)
                        
#                         all_patches_b = torch.cat(region_patches_list, dim=0)
                
#                         nodes_b, V, E = self.build_hypergraph_from_regions(
#                             text_nodes_b, all_patches_b, current_batch_region_indices, 
#                             region_patches_list, threshold=0.6
#                         )
                        
#                         if V is not None and E is not None:
#                             self.hgraph_shared.V = V
#                             self.hgraph_shared.E = E
#                             nodes_b = self.hgraph_shared(nodes_b)
#                     else:
#                         region_patches_list = []
#                         for region_tokens in part_patch_tokens:
#                             if region_tokens.dim() == 3 and region_tokens.shape[0] == b_s:
#                                 region_patches_list.append(region_tokens[b])
#                             else:
#                                 region_patches_list.append(region_tokens[b] if region_tokens.shape[0] > b else region_tokens)
                        
#                         nodes_b = torch.cat(region_patches_list, dim=0)

#                     enhanced_batch.append(nodes_b)

#                 max_nodes = max([nodes_b.shape[0] for nodes_b in enhanced_batch])
#                 padded_batch = []
#                 for nodes_b in enhanced_batch:
#                     if nodes_b.shape[0] < max_nodes:
#                         pad_size = max_nodes - nodes_b.shape[0]
#                         pad = torch.zeros(pad_size, nodes_b.shape[1], device=nodes_b.device)
#                         nodes_b = torch.cat([nodes_b, pad], dim=0)
#                     padded_batch.append(nodes_b.unsqueeze(0))
#                 enhanced_features = torch.cat(padded_batch, dim=0)

#         x = torch.cat([enhanced_features, self.visual_embed(clip_image_features.float())], dim=1)

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







# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from clip import clip
# from models.vit import *
# from config import argument_parser
# from .UniGNN import UniGNN
# from torch_geometric.utils import dense_to_sparse
# parser = argument_parser()
# args = parser.parse_args()

# class TransformerClassifier(nn.Module):
#     def __init__(self, clip_model, attr_num, attributes, dim=768,
#                  pretrain_path='/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb6/DATA/wushujuan/PromptPAR/jx_vit_base_p16_224-80ecf9dd.pth'):
#         super().__init__()
#         self.attr_num = attr_num
#         self.dim = dim
#         self.word_embed = nn.Linear(clip_model.visual.output_dim, dim)
#         self.visual_embed = nn.Linear(clip_model.visual.output_dim, dim)

#         vit = vit_base()
#         vit.load_param(pretrain_path)
#         self.norm = vit.norm
#         self.weight_layer = nn.ModuleList([nn.Linear(dim, 1) for _ in range(self.attr_num)])
#         self.text = clip.tokenize(attributes).to("cuda")
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
#             nhid = 1024
#             nhead = 4
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
       
#         data_path = '/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb6/DATA/wushujuan/PromptPAR/dataset/rapv1_unified_clip_features.pt'
#         data = torch.load(data_path, weights_only=False)

#         features = data['features'].to("cuda")      
#         labels = data['labels']
       
#         # 全局超图 V/E

#         edge_index, _ = dense_to_sparse(labels.float())
#         V = edge_index[0].to("cuda")
#         E = edge_index[1].to("cuda")

#         self.hgraph_global = UniGNN(
#             args=gnn_args,
#             nfeat=features.shape[1],
#             nhid=gnn_args.nhid,
#             nclass=dim,
#             nlayer=gnn_args.nlayer,
#             nhead=gnn_args.nhead,
#             V=V,
#             E=E
#         ).to("cuda")
#         with torch.no_grad():
#             self.global_hg = self.hgraph_global(features).detach()
#     def extract_region_texts_from_logits(self, logits_per_image, text_features, threshold=0.5):
#         """
#         局部文本优先，全局文本不能重复
#         """
#         batch_size, total_regions, attr_num = logits_per_image.shape
#         used_text_indices = [set() for _ in range(batch_size)] 
#         region_text_indices_filtered = []

#         for region_idx in range(total_regions):
#             region_indices_batch = []
#             region_logits = logits_per_image[:, region_idx, :]
#             for batch_idx in range(batch_size):
#                 sample_logits = region_logits[batch_idx]
#                 high_sim_indices = torch.where(sample_logits > threshold)[0].tolist()
#                 new_indices = [idx for idx in high_sim_indices if idx not in used_text_indices[batch_idx]]
#                 used_text_indices[batch_idx].update(new_indices)
#                 region_indices_batch.append(torch.tensor(new_indices, device=text_features.device, dtype=torch.long))
#             region_text_indices_filtered.append(region_indices_batch)
#         return region_text_indices_filtered

#     def build_hypergraph_from_regions(self, text_nodes_b, all_patches_b, filtered_region_indices, region_patches_list, threshold=0.6):
#         if len(text_nodes_b) == 0 or len(region_patches_list) == 0:
#             return None, None, None

#         nodes = torch.cat([text_nodes_b, all_patches_b], dim=0)
#         num_text_nodes = text_nodes_b.shape[0]
        
#         V_list, E_list = [], []
#         hyperedge_idx = 0
#         text_idx_offset = 0

#         for region_idx, region_patches in enumerate(region_patches_list):
#             region_text_indices = filtered_region_indices[region_idx]
            
#             if len(region_text_indices) == 0 or region_patches.shape[0] == 0:
#                 continue
#             region_text_features = text_nodes_b[text_idx_offset:text_idx_offset+len(region_text_indices)]
#             text_idx_offset += len(region_text_indices)
        
#             text_norm = F.normalize(region_text_features.float(), dim=-1)
#             patch_norm = F.normalize(region_patches.float(), dim=-1)
#             similarity_matrix = torch.mm(text_norm, patch_norm.t())
            
#             for text_local_idx, text_global_idx in enumerate(region_text_indices):
#                 similarities = similarity_matrix[text_local_idx] 
#                 high_sim_patches = torch.where(similarities > threshold)[0]
                
#                 if len(high_sim_patches) > 0:
#                     text_node_idx = text_global_idx.item() if hasattr(text_global_idx, 'item') else text_global_idx
#                     V_list.append(text_node_idx)
#                     E_list.append(hyperedge_idx)
#                     region_patch_start = sum([rp.shape[0] for rp in region_patches_list[:region_idx]])
#                     for patch_local_idx in high_sim_patches:
#                         patch_global_idx = num_text_nodes + region_patch_start + patch_local_idx.item()
#                         V_list.append(patch_global_idx)
#                         E_list.append(hyperedge_idx)
                    
#                     hyperedge_idx += 1
        
#         if len(V_list) == 0:
#             return nodes, None, None
            
#         V = torch.tensor(V_list, dtype=torch.long, device=nodes.device)
#         E = torch.tensor(E_list, dtype=torch.long, device=nodes.device)
        
#         return nodes, V, E

#     def forward(self, imgs, clip_model):
#         b_s = imgs.shape[0]
#         clip_image_features, all_class, attenmap, part_patch_tokens = clip_model.visual(imgs.type(clip_model.dtype))
        
#         text_features = clip_model.encode_text(self.text).to("cuda").float()

#         enhanced_features = None
#         final_similarity = None
#         if args.use_div:
#             final_similarity, logits_per_image = clip_model.forward_aggregate(all_class, text_features)
#             filtered_region_indices = self.extract_region_texts_from_logits(logits_per_image, text_features, threshold=0.6)

#             if part_patch_tokens is not None:
#                 enhanced_batch = []
#                 for b in range(b_s):
#                     text_nodes_list = []
#                     current_batch_region_indices = []
#                     for region_idx in range(len(filtered_region_indices)):
#                         region_indices = filtered_region_indices[region_idx][b]
#                         current_batch_region_indices.append(region_indices)
#                         if len(region_indices) > 0:
#                             text_nodes_list.append(text_features[region_indices])
                    
#                     if len(text_nodes_list) > 0:
#                         text_nodes_b = torch.cat(text_nodes_list, dim=0)  # [num_selected_texts, 768]
# s
#                         region_patches_list = [region_tokens[b] for region_tokens in part_patch_tokens]  #  [patch_num, 768]
#                         all_patches_b = torch.cat(region_patches_list, dim=0)  # [total_patches, 768]

#                         nodes_b, V, E = self.build_hypergraph_from_regions(
#                             text_nodes_b, all_patches_b, current_batch_region_indices, 
#                             region_patches_list, threshold=0.6
#                         )
                       
#                         if V is not None and E is not None:
#                             self.hgraph_shared.V = V
#                             self.hgraph_shared.E = E
#                             nodes_b = self.hgraph_shared(nodes_b)  #  [total_nodes, 768]
#                     else:
#                         region_patches_list = [region_tokens[b] for region_tokens in part_patch_tokens]
#                         nodes_b = torch.cat(region_patches_list, dim=0)  # [total_patches, 768]

#                     enhanced_batch.append(nodes_b)  # nodes_b 是 [num_nodes, 768]

#                 max_nodes = max([nodes_b.shape[0] for nodes_b in enhanced_batch])
#                 padded_batch = []
#                 for nodes_b in enhanced_batch:
#                     if nodes_b.shape[0] < max_nodes:
#                         pad_size = max_nodes - nodes_b.shape[0]
#                         pad = torch.zeros(pad_size, nodes_b.shape[1], device=nodes_b.device)
#                         nodes_b = torch.cat([nodes_b, pad], dim=0)
#                     padded_batch.append(nodes_b.unsqueeze(0))  # [1, max_nodes, 768]
#                 enhanced_features = torch.cat(padded_batch, dim=0)  # [batch_size, max_nodes, 768]
       
#         cls_token = all_class[:, 0].float()         
#         global_hg = self.global_hg.float()         

#         sim_matrix = torch.matmul(
#             F.normalize(cls_token, dim=-1),
#             F.normalize(self.global_hg, dim=-1).t()
#         )  

#         with torch.no_grad():
#             topk_values, topk_indices = torch.topk(sim_matrix, k=10, dim=1)

#         topk_global_features = self.global_hg[topk_indices]   # [B, 10, 768]

#         x = torch.cat([
#             enhanced_features,   
#             topk_global_features ,
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
# from .UniGNN import UniGNN
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

#         self.text_features = None
#         self.register_buffer('global_hg', None)
        
#         self.word_embed = nn.Linear(clip_model.visual.output_dim, dim)
#         self.visual_embed = nn.Linear(clip_model.visual.output_dim, dim)

#         vit = vit_base()
#         vit.load_param(pretrain_path)
#         self.norm = vit.norm
#         self.weight_layer = nn.ModuleList([nn.Linear(dim, 1) for _ in range(self.attr_num)])
        
#         text_file_path = '/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb6/DATA/wushujuan/PromptPAR/dataset/hy/rapv1.txt'
#         with open(text_file_path, 'r', encoding='utf-8') as f:
#             self.all_texts = [line.strip() for line in f if line.strip()]

#         self.all_tokens = clip.tokenize(self.all_texts).to("cuda")
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
#             nlayer = 1
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

#         self._preload_global_data()

#     def _preload_global_data(self):
#         data_path = '/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb6/DATA/wushujuan/PromptPAR/dataset/hy/rapv1_unified_clip_features.pt'
#         data = torch.load(data_path, weights_only=False)

#         self.global_features = data['features']
#         self.global_labels = data['labels']

#         edge_index, _ = dense_to_sparse(self.global_labels.float())
#         self.global_V = edge_index[0]
#         self.global_E = edge_index[1]
        
#         self.data_loaded = True

#     def _setup_global_gnn(self, device):

#         if self.global_hg is None:
#             features = self.global_features.to(device)
#             V = self.global_V.to(device)
#             E = self.global_E.to(device)

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

#     def _precompute_text_features(self, clip_model):

#         if self.text_features is None:
#             with torch.no_grad():
#                 self.text_features = clip_model.encode_text(self.all_tokens).float()

#     def build_unified_hypergraph_batch_optimized(self, patch_tokens_batch, text_features, threshold=0.5):

#         batch_size, num_patches, dim = patch_tokens_batch.shape
#         num_texts = text_features.shape[0]

#         if patch_tokens_batch.dtype != text_features.dtype:
#             patch_tokens_batch = patch_tokens_batch.to(text_features.dtype)
 
#         patch_norm = F.normalize(patch_tokens_batch, dim=-1)
#         text_norm = F.normalize(text_features, dim=-1)
#         sim_matrix = torch.matmul(patch_norm, text_norm.T)  # [batch_size, num_patches, num_texts]

#         enhanced_batch = []

#         for b in range(batch_size):
#             patch_tokens = patch_tokens_batch[b]  # [num_patches, dim]
#             sim_matrix_b = sim_matrix[b]  # [num_patches, num_texts]
            
#             all_nodes = torch.cat([patch_tokens, text_features], dim=0)

#             valid_connections = sim_matrix_b > threshold
#             patch_indices, text_indices = torch.where(valid_connections)
            
#             if len(patch_indices) == 0:
#                 enhanced_batch.append(all_nodes.unsqueeze(0))
#                 continue

#             V_list = []
#             E_list = []
#             hyperedge_counter = 0

#             for text_idx in range(num_texts):
#                 connected_patches = patch_indices[text_indices == text_idx]
#                 if len(connected_patches) > 0:
#                     V_list.extend(connected_patches.tolist())
#                     E_list.extend([hyperedge_counter] * len(connected_patches))

#                     text_global_idx = num_patches + text_idx
#                     V_list.append(text_global_idx)
#                     E_list.append(hyperedge_counter)
#                     hyperedge_counter += 1
            
#             if len(V_list) == 0:
#                 enhanced_batch.append(all_nodes.unsqueeze(0))
#                 continue
            
#             V = torch.tensor(V_list, device=all_nodes.device, dtype=torch.long)
#             E = torch.tensor(E_list, device=all_nodes.device, dtype=torch.long)

#             self.hgraph_shared.V = V
#             self.hgraph_shared.E = E
#             enhanced_nodes = self.hgraph_shared(all_nodes.float())
#             enhanced_batch.append(enhanced_nodes.unsqueeze(0))
        
#         return enhanced_batch
    
#     def forward(self, imgs, clip_model):
#         device = imgs.device
#         b_s = imgs.shape[0]

#         self._setup_global_gnn(device)
#         self._precompute_text_features(clip_model)

#         clip_output = clip_model.visual(imgs.type(clip_model.dtype))
#         clip_image_features, all_class, attnmap, part_patch_tokens, part_indices_info = clip_output

#         enhanced_features = None
#         final_similarity = None

#         if args.use_div and part_patch_tokens is not None:
#             global_patches_batch = part_patch_tokens[0]  # [batch_size, 256, dim]
#             final_similarity, logits_per_image = clip_model.forward_aggregate(all_class, self.text_features)
        
#             enhanced_batch = self.build_unified_hypergraph_batch_optimized(
#                 global_patches_batch, self.text_features, threshold=0.6
#             )
    
#             if enhanced_batch:
#                 node_counts = torch.tensor([feat.shape[1] for feat in enhanced_batch], device=device)
#                 min_nodes = int(torch.min(node_counts).item())  

#                 padded_batch = []
#                 for feat in enhanced_batch:
#                     if feat.shape[1] > min_nodes:
#                         indices = torch.randperm(feat.shape[1], device=device)[:min_nodes]
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

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from clip import clip
# from models.vit import *
# from config import argument_parser
# from .UniGNN import UniGNN
# from torch_geometric.utils import dense_to_sparse
# import json

# parser = argument_parser()
# args = parser.parse_args()


# class TransformerClassifier(nn.Module):
#     def __init__(self, clip_model, attr_num, attributes,
#                  dim=512,
#                  pretrain_path='/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb6/DATA/wushujuan/PromptPAR/jx_vit_base_p16_224-80ecf9dd.pth'):
#         super().__init__()
#         self.attr_num = attr_num
#         self.dim = dim
#         self.region_text_features = None
#         self.text_ranges = None
#         self.register_buffer('global_hg', None)

#         # 投影层，把 ViT 输出对齐 CLIP 维度
#         if clip_model.visual.output_dim == 768:   # 用自带ViT (768d)
#             self.proj_to_clip = nn.Linear(768, 512)
#         else:  # CLIP ViT-B/16 已经是 512d
#             self.proj_to_clip = nn.Identity()

#         self.word_embed = nn.Linear(clip_model.visual.output_dim, dim)
#         self.visual_embed = nn.Linear(clip_model.visual.output_dim, dim)

#         vit = vit_base()
#         vit.load_param(pretrain_path)
#         self.norm = vit.norm

#                 # ===== 在初始化里修改 =====
#         # 原来
#         # self.weight_layer = nn.ModuleList([nn.Linear(dim, 1) for _ in range(self.attr_num)])
#         # 改成
#         weight_dim = 768 if args.use_mm_former else dim
#         self.weight_layer = nn.ModuleList([nn.Linear(weight_dim, 1) for _ in range(self.attr_num)])


#         # 加载属性文本
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
#             # ===== 新增投影层，把512 -> 768 =====
#             self.mm_proj = nn.Linear(512, 768)

#         # UniGNN 参数
#         class UniGNNArgs:
#             model_name = 'UniGAT'
#             first_aggregate = 'mean'
#             second_aggregate = 'sum'
#             add_self_loop = False
#             use_norm = True
#             activation = 'relu'
#             nlayer = 1
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
#             data_path = '/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb6/DATA/wushujuan/PromptPAR/dataset/hy/rapv1_unified_clip_features_512.pt'
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
#         all_nodes = [global_patches]
#         text_start_idx = global_patches.shape[0]
#         text_count = 0
#         region_names = ['body', 'head', 'upper', 'lower', 'foot']
#         for region_name in region_names:
#             texts = region_texts_dict[region_name]
#             all_nodes.append(texts)
#             text_count += texts.shape[0]
#         all_nodes = torch.cat(all_nodes, dim=0)

#         V_list, E_list = [], []
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

#         enhanced_batch = []
#         for b in range(b_s):
#             global_patches = part_patch_tokens[0][b]
#             all_nodes, V, E = self.build_unified_hypergraph(global_patches, part_indices_info, self.region_text_features, threshold=0.6)
#             if V is not None and E is not None:
#                 self.hgraph_shared.V = V
#                 self.hgraph_shared.E = E
#                 enhanced_nodes = self.hgraph_shared(all_nodes.float())
#                 enhanced_batch.append(enhanced_nodes.unsqueeze(0))
#             else:
#                 enhanced_batch.append(all_nodes.unsqueeze(0))

#         if enhanced_batch:
#             node_counts = [feat.shape[1] for feat in enhanced_batch]
#             min_nodes = int(torch.min(torch.tensor(node_counts)).item())
#             padded_batch = []
#             for feat in enhanced_batch:
#                 if feat.shape[1] > min_nodes:
#                     indices = torch.randperm(feat.shape[1], device=feat.device)[:min_nodes]
#                     feat = feat[:, indices, :]
#                 padded_batch.append(feat)
#             enhanced_features = torch.cat(padded_batch, dim=0)

#         # === 关键修改：cls_token 投影到512，与global_hg对齐 ===
#         cls_token = all_class[:, 0].float()  # (B, 768)
#         cls_token_proj = self.proj_to_clip(cls_token)   # (B, 512)

#         global_hg = self.global_hg.float()              # (N, 512)
#         global_hg_norm = F.normalize(global_hg, dim=-1)
#         clip_img_norm = F.normalize(cls_token_proj, dim=-1)

#         sim_matrix = torch.einsum("bd,nd->bn", clip_img_norm, global_hg_norm)  # [B, N]

#         with torch.no_grad():
#             topk_values, topk_indices = torch.topk(sim_matrix, k=10, dim=1)
#             topk_global_features = global_hg[topk_indices]

#         x = torch.cat([
#             enhanced_features,
#             topk_global_features,
#             self.visual_embed(clip_image_features.float())
#         ], dim=1)

#         if args.use_mm_former:
#             # ===== 投影到768送入ViT block =====
#             x = self.mm_proj(x)
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



# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from clip import clip
# from models.vit import *
# from config import argument_parser
# from .UniGNN import UniGNN
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
        
#         json_path = '/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb6/DATA/wushujuan/PromptPAR/dataset/rapv1.json'
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
#             nhid = 1024
#             nhead = 4
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
#             data_path = '/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb6/DATA/wushujuan/PromptPAR/dataset/rapv1_unified_clip_features.pt'
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
            
#             # 预计算文本索引范围
#             self.text_ranges = {}
#             start_idx = 0
#             for region in ['body', 'head', 'upper', 'lower', 'foot']:
#                 if region in self.region_text_features:
#                     num_texts = self.region_text_features[region].shape[0]
#                     self.text_ranges[region] = (start_idx, start_idx + num_texts)
#                     start_idx += num_texts

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
                    
#                     # 只有当找到相似的patch时才创建超边
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

#             text_ranges = self.text_ranges

#             filtered_region_texts = {}
#             for i, region in enumerate(region_names):
#                 region_sim = logits_per_image[:, i, :]
#                 start, end = text_ranges[region]
#                 region_text_sim = region_sim[:, start:end]
                
#                 filtered_texts_list = []
#                 for b in range(b_s):
#                     sim_values = region_text_sim[b]
#                     num_texts = len(sim_values)
#                     num_to_keep = max(1, int(num_texts * 0.6))
                    
#                     _, topk_indices = torch.topk(sim_values, k=num_to_keep)
#                     topk_texts = self.region_text_features[region][topk_indices]
#                     filtered_texts_list.append(topk_texts)
                
#                 filtered_region_texts[region] = filtered_texts_list
            
#             enhanced_batch = []
            
#             for b in range(b_s):
#                 global_patches = part_patch_tokens[0][b]
                
#                 current_texts_dict = {}
#                 for region in region_names:
#                     current_texts_dict[region] = filtered_region_texts[region][b]
                
#                 all_nodes, V, E = self.build_unified_hypergraph(
#                     global_patches, part_indices_info, current_texts_dict, threshold=0.6
#                 )
                
#                 if V is not None and E is not None:
#                     self.hgraph_shared.V = V
#                     self.hgraph_shared.E = E
#                     enhanced_nodes = self.hgraph_shared(all_nodes.float())
#                     enhanced_batch.append(enhanced_nodes.unsqueeze(0))
#                 else:
#                     enhanced_batch.append(all_nodes.unsqueeze(0))
            
#             if enhanced_batch:
#                 max_nodes = max([feat.shape[1] for feat in enhanced_batch])
#                 padded_batch = []
#                 for feat in enhanced_batch:
#                     if feat.shape[1] < max_nodes:
#                         pad_size = max_nodes - feat.shape[1]
#                         pad = torch.zeros(feat.shape[0], pad_size, feat.shape[2], device=feat.device)
#                         feat = torch.cat([feat, pad], dim=1)
#                     padded_batch.append(feat)
                
#                 enhanced_features = torch.cat(padded_batch, dim=0)

   

#         cls_token = all_class[:, 0].float()
#         global_hg = self.global_hg.float()

#         sim_matrix = torch.matmul(
#             F.normalize(cls_token, dim=-1),
#             F.normalize(self.global_hg, dim=-1).t()
#         )

#         with torch.no_grad():
#             topk_values, topk_indices = torch.topk(sim_matrix, k=10, dim=1)

#         topk_global_features = self.global_hg[topk_indices]

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



# import torch.nn as nn
# import torch
# from clip import clip
# from models.vit import *
# from config import argument_parser
# parser = argument_parser()
# args = parser.parse_args()
# class TransformerClassifier(nn.Module):
#     def __init__(self, clip_model, attr_num, attributes, dim=768, pretrain_path='/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb6/DATA/wushujuan/PromptPAR/jx_vit_base_p16_224-80ecf9dd.pth'):
#         super().__init__()
#         super().__init__()
#         self.attr_num = attr_num
#         self.word_embed = nn.Linear(clip_model.visual.output_dim, dim)
#         self.visual_embed = nn.Linear(clip_model.visual.output_dim, dim)
#         vit = vit_base()
#         vit.load_param(pretrain_path)
#         self.norm = vit.norm
#         self.weight_layer = nn.ModuleList([nn.Linear(dim, 1) for i in range(self.attr_num)])
#         self.dim = dim
#         self.text = clip.tokenize(attributes).to("cuda")
#         self.bn = nn.BatchNorm1d(self.attr_num)
#         fusion_len = self.attr_num + 257 + args.vis_prompt
#         if not args.use_mm_former :
#             print('Without MM-former, Using MLP Instead')
#             self.linear_layer = nn.Linear(fusion_len, self.attr_num)
#         else:
#             self.blocks = vit.blocks[-args.mm_layers:]
#     def forward(self,imgs,clip_model):
#         b_s=imgs.shape[0]
#         clip_image_features, all_class, attnmap, part_patch_tokens, part_indices_info = clip_model.visual(imgs.type(clip_model.dtype))
#         text_features = clip_model.encode_text(self.text).to("cuda").float()
#         if args.use_div:
#             final_similarity,logits_per_image = clip_model.forward_aggregate(all_class,text_features)
#         else : 
#             final_similarity = None
#         textual_features = self.word_embed(text_features).expand(b_s, self.attr_num, self.dim)
#         x = torch.cat([self.visual_embed(clip_image_features.float())], dim=1)
        
#         if args.use_mm_former:
#             for blk in self.blocks:
#                 x = blk(x)
#         else :# using linear layer fusion
#             x = x.permute(0, 2, 1)
#             x= self.linear_layer(x)
#             x = x.permute(0, 2, 1)
            
#         x = self.norm(x)
#         logits = torch.cat([self.weight_layer[i](x[:, i, :]) for i in range(self.attr_num)], dim=1)
#         bn_logits = self.bn(logits)
        
        
#         return bn_logits,final_similarity