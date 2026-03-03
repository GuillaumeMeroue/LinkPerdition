import torch
import torch.nn as nn
import torch.nn.functional as F

from kge.utils import set_seed
from kge.utils import get_init_function


MODELS = ['TransE', 'DistMult', 'ComplEx', 'RotatE', 'ConvE', 'RGCN', 'Transformer']

class KGEModel(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim, seed_forward, use_inverse=False):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.use_inverse = use_inverse
        self.seed_forward = seed_forward
        self.forward_count = 0

    def forward(self, heads, relations, tails, score_mode="triple"):
        set_seed(self.seed_forward + self.forward_count)
        self.forward_count += 1
        # print(f"Forward count: {self.forward_count}")
        # If using inverse relations and multi_heads mode, transform to multi_tails with inverse relations
        if self.use_inverse and score_mode == "multi_heads":
            # Compute inverse relation indices
            inverse_relations = (relations + self.num_relations // 2) % self.num_relations
            # Swap heads and tails, switch to multi_tails
            return self._forward_internal(tails, inverse_relations, heads, score_mode="multi_tails")
        # Otherwise, call the subclass implementation
        return self._forward_internal(heads, relations, tails, score_mode=score_mode)

    def _forward_internal(self, heads, relations, tails, score_mode="triple"):
        raise NotImplementedError


    def get_use_inverse(self):
        return getattr(self, 'use_inverse', False)

    def set_use_inverse(self, value: bool):
        self.use_inverse = value


class TransE(KGEModel):
    def __init__(self, num_entities, num_relations, embedding_dim, seed_forward, seed_init, use_inverse, transE_norm, dropout_entity, dropout_relation, init_function):
        super().__init__(num_entities, num_relations, embedding_dim, seed_forward, use_inverse=use_inverse)
        set_seed(seed_init)
        self.entity_emb = nn.Embedding(num_entities, embedding_dim)
        self.relation_emb = nn.Embedding(num_relations, embedding_dim)
        self._norm = transE_norm
        self.entity_dropout = nn.Dropout(dropout_entity)
        self.relation_dropout = nn.Dropout(dropout_relation)
        self.seed_init = seed_init
        self.init_function = get_init_function(init_function)
        self.reset_parameters()


    def reset_parameters(self):
        set_seed(self.seed_init)
        print("Resetting parameters with seed:", self.seed_init, "and init function:", self.init_function.__name__)
        self.init_function(self.entity_emb.weight.data)
        self.init_function(self.relation_emb.weight.data)
        print("hash of parameters:", hash(self.entity_emb.weight.data), hash(self.relation_emb.weight.data))


    def score_emb(self, h_emb, r_emb, t_emb, score_mode="triple"):
        # h_emb: [batch, emb_dim] or [batch, 1+num_neg, emb_dim]
        # score_mode: 'triple', 'multi_tails', 'multi_heads'
        n = r_emb.size(0)
        h_emb = self.entity_dropout(h_emb)
        r_emb = self.relation_dropout(r_emb)
        t_emb = self.entity_dropout(t_emb)
        if score_mode == "triple":
            out = -torch.norm(h_emb + r_emb - t_emb, p=self._norm, dim=1)
        elif score_mode == "multi_tails":
            # h_emb, r_emb: [batch, emb_dim], t_emb: [batch, 1+num_neg, emb_dim]
            left = (h_emb + r_emb).unsqueeze(1)
            out = -torch.cdist(left, t_emb, p=self._norm, compute_mode="donot_use_mm_for_euclid_dist").squeeze(1)
        elif score_mode == "multi_heads":
            # t_emb, r_emb: [batch, emb_dim], h_emb: [batch, 1+num_neg, emb_dim]
            left = (t_emb - r_emb).unsqueeze(1)
            out = -torch.cdist(left, h_emb, p=self._norm, compute_mode="donot_use_mm_for_euclid_dist").squeeze(1)
        else:
            raise ValueError(f"Unknown score_mode: {score_mode}")
        return out.view(n, -1)

    def _forward_internal(self, heads, relations, tails, score_mode="triple"):
        h = self.entity_emb(heads)
        r = self.relation_emb(relations)
        t = self.entity_emb(tails)
        return self.score_emb(h, r, t, score_mode=score_mode)


class DistMult(KGEModel):
    def __init__(self, num_entities, num_relations, embedding_dim, seed_forward, seed_init, use_inverse, dropout_entity, dropout_relation, init_function):
        super().__init__(num_entities, num_relations, embedding_dim, seed_forward, use_inverse=use_inverse)
        set_seed(seed_init)
        self.entity_emb = nn.Embedding(num_entities, embedding_dim)
        self.relation_emb = nn.Embedding(num_relations, embedding_dim)
        self.entity_dropout = nn.Dropout(dropout_entity)
        self.relation_dropout = nn.Dropout(dropout_relation)
        self.seed_init = seed_init
        self.init_function = get_init_function(init_function)
        self.reset_parameters()

    def reset_parameters(self):
        set_seed(self.seed_init)
        print("Resetting parameters with seed:", self.seed_init)
        self.init_function(self.entity_emb.weight.data)
        self.init_function(self.relation_emb.weight.data)
        print("hash of parameters:", hash(self.entity_emb.weight.data), hash(self.relation_emb.weight.data))

    def score_emb(self, h_emb, r_emb, t_emb, score_mode="triple"):
        # h_emb: [batch, emb_dim] or [batch, 1+num_neg, emb_dim]
        # score_mode: 'triple', 'multi_tails', 'multi_heads'
        n = r_emb.size(0)
        h_emb = self.entity_dropout(h_emb)
        r_emb = self.relation_dropout(r_emb)
        t_emb = self.entity_dropout(t_emb)
        if score_mode == "triple":
            out = (h_emb * r_emb * t_emb).sum(dim=1)
        elif score_mode == "multi_tails":
            # left: [batch, emb_dim]
            left = h_emb * r_emb
            # right: [batch, 1+num_neg, emb_dim]
            out = torch.bmm(left.unsqueeze(1), t_emb.transpose(1, 2)).squeeze(1)
        elif score_mode == "multi_heads":
            # left: [batch, emb_dim]
            left = t_emb * r_emb
            # right: [batch, 1+num_neg, emb_dim]
            out = torch.bmm(left.unsqueeze(1), h_emb.transpose(1, 2)).squeeze(1)
        else:
            raise ValueError(f"Unknown score_mode: {score_mode}")
        return out.view(n, -1)

    def _forward_internal(self, heads, relations, tails, score_mode="triple"):
        h = self.entity_emb(heads)
        r = self.relation_emb(relations)
        t = self.entity_emb(tails)
        return self.score_emb(h, r, t, score_mode=score_mode)


class ComplEx(KGEModel):
    """ComplEx model: Complex embeddings for simple link prediction.
    
    Reference: Théo Trouillon et al. "Complex Embeddings for Simple Link Prediction." ICML 2016.
    
    Embeddings are stored as real vectors of size 2*embedding_dim, where the first half
    represents the real part and the second half represents the imaginary part.
    The score function computes Re(<h, r, conj(t)>) where conj is complex conjugate.
    """
    def __init__(self, num_entities, num_relations, embedding_dim, seed_forward, seed_init, use_inverse, dropout_entity, dropout_relation, init_function):
        super().__init__(num_entities, num_relations, embedding_dim, seed_forward, use_inverse=use_inverse)
        set_seed(seed_init)
        # Embeddings are 2x size to store real and imaginary parts
        # First half: real part, second half: imaginary part
        self.entity_emb = nn.Embedding(num_entities, embedding_dim)
        self.relation_emb = nn.Embedding(num_relations, embedding_dim)
        self.entity_dropout = nn.Dropout(dropout_entity)
        self.relation_dropout = nn.Dropout(dropout_relation)
        self.seed_init = seed_init
        self.init_function = get_init_function(init_function)
        self.reset_parameters()

    def reset_parameters(self):
        set_seed(self.seed_init)
        print("Resetting parameters with seed:", self.seed_init)
        self.init_function(self.entity_emb.weight.data)
        self.init_function(self.relation_emb.weight.data)
        print("hash of parameters:", hash(self.entity_emb.weight.data), hash(self.relation_emb.weight.data))

    def score_emb(self, h_emb, r_emb, t_emb, score_mode="triple"):
        """Compute ComplEx scores: Re(<h, r, conj(t)>)
        
        Following LibKGE's fast implementation using Hadamard products.
        Embeddings are split into real (first half) and imaginary (second half) parts.
        
        Args:
            h_emb: [batch, 2*emb_dim] or [batch, 1+num_neg, 2*emb_dim]
            r_emb: [batch, 2*emb_dim]
            t_emb: [batch, 2*emb_dim] or [batch, 1+num_neg, 2*emb_dim]
            score_mode: 'triple', 'multi_tails', 'multi_heads'
        """
        n = r_emb.size(0)
        h_emb = self.entity_dropout(h_emb)
        r_emb = self.relation_dropout(r_emb)
        t_emb = self.entity_dropout(t_emb)
        
        # Split embeddings into real and imaginary parts
        # For h_emb: first half is real, second half is imaginary
        if score_mode == "triple":
            # h_emb, r_emb, t_emb: [batch, 2*emb_dim]
            r_emb_re, r_emb_im = r_emb.chunk(2, dim=1)
            t_emb_re, t_emb_im = t_emb.chunk(2, dim=1)
            
            # Fast ComplEx scoring as in LibKGE (Eq. 11 of paper)
            # Compute Re(<h, r, conj(t)>) efficiently
            h_all = torch.cat([h_emb, h_emb], dim=1)  # [batch, 4*emb_dim]: re, im, re, im
            r_all = torch.cat([r_emb_re, r_emb, -r_emb_im], dim=1)  # [batch, 4*emb_dim]: re, re, im, -im
            t_all = torch.cat([t_emb, t_emb_im, t_emb_re], dim=1)  # [batch, 4*emb_dim]: re, im, im, re
            
            out = (h_all * r_all * t_all).sum(dim=1)
            
        elif score_mode == "multi_tails":
            # h_emb, r_emb: [batch, 2*emb_dim]
            # t_emb: [batch, 1+num_neg, 2*emb_dim]
            r_emb_re, r_emb_im = r_emb.chunk(2, dim=1)
            t_emb_re, t_emb_im = t_emb.chunk(2, dim=2)  # Split along last dimension
            
            # Expand h and r for broadcasting
            h_all = torch.cat([h_emb, h_emb], dim=1)  # [batch, 4*emb_dim]
            r_all = torch.cat([r_emb_re, r_emb, -r_emb_im], dim=1)  # [batch, 4*emb_dim]
            t_all = torch.cat([t_emb, t_emb_im, t_emb_re], dim=2)  # [batch, 1+num_neg, 4*emb_dim]
            
            # Compute (h * r) first, then batch matrix multiply with t
            left = h_all * r_all  # [batch, 4*emb_dim]
            out = torch.bmm(left.unsqueeze(1), t_all.transpose(1, 2)).squeeze(1)
            
        elif score_mode == "multi_heads":
            # h_emb: [batch, 1+num_neg, 2*emb_dim]
            # r_emb, t_emb: [batch, 2*emb_dim]
            r_emb_re, r_emb_im = r_emb.chunk(2, dim=1)
            h_emb_re, h_emb_im = h_emb.chunk(2, dim=2)  # Split along last dimension
            
            # For multi_heads, we compute Re(<h, r, conj(t)>) for multiple h
            # This requires adapting the formula
            h_all = torch.cat([h_emb, h_emb], dim=2)  # [batch, 1+num_neg, 4*emb_dim]
            r_all = torch.cat([r_emb_re, r_emb, -r_emb_im], dim=1)  # [batch, 4*emb_dim]
            t_all = torch.cat([t_emb, t_emb.chunk(2, dim=1)[1], t_emb.chunk(2, dim=1)[0]], dim=1)  # [batch, 4*emb_dim]
            
            # Compute (r * t) first
            right = r_all * t_all  # [batch, 4*emb_dim]
            out = torch.bmm(h_all, right.unsqueeze(2)).squeeze(2)
            
        else:
            raise ValueError(f"Unknown score_mode: {score_mode}")
        
        return out.view(n, -1)

    def _forward_internal(self, heads, relations, tails, score_mode="triple"):
        h = self.entity_emb(heads)
        r = self.relation_emb(relations)
        t = self.entity_emb(tails)
        return self.score_emb(h, r, t, score_mode=score_mode)


class RotatE(KGEModel):
    """RotatE model: Rotation-based embeddings in complex space.
    
    Reference: Zhiqing Sun et al. "RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space." ICLR 2019.
    
    Entities are represented as complex vectors (stored as 2*embedding_dim with real/imaginary parts).
    Relations are rotations in complex space, stored as angles (embedding_dim dimensions).
    Score function: -|| h ⊙ r - t || where ⊙ is complex Hadamard product and r = e^(i*theta).
    """
    def __init__(self, num_entities, num_relations, embedding_dim, seed_forward, seed_init, use_inverse, 
                 dropout_entity, dropout_relation, init_function, rotate_norm=2, normalize_phases=True):
        super().__init__(num_entities, num_relations, embedding_dim, seed_forward, use_inverse=use_inverse)
        set_seed(seed_init)
        
        # Entity embeddings are complex: embedding_dim (real + imaginary parts)
        self.entity_emb = nn.Embedding(num_entities, embedding_dim)
        
        # Relations are stored as angles (phases): embedding_dim dimensions
        # Each angle represents a rotation in the complex plane
        self.relation_emb = nn.Embedding(num_relations, embedding_dim // 2)
        
        self.entity_dropout = nn.Dropout(dropout_entity)
        self.relation_dropout = nn.Dropout(dropout_relation)
        self.seed_init = seed_init
        self.init_function = get_init_function(init_function)
        self._norm = rotate_norm
        self._normalize_phases = normalize_phases
        self.reset_parameters()

    def reset_parameters(self):
        import math
        set_seed(self.seed_init)
        print("Resetting parameters with seed:", self.seed_init)
        self.init_function(self.entity_emb.weight.data)
        # Initialize relation phases uniformly in [-pi, pi] with seed control
        self.relation_emb.weight.data.uniform_(-math.pi, math.pi)
        print("hash of parameters:", hash(self.entity_emb.weight.data), hash(self.relation_emb.weight.data))

    @torch.no_grad()
    def normalize_phases(self):
        """Normalize relation phases to lie in [-pi, pi].
        
        This is adapted from LibKGE's implementation. The 'hack' refers to direct
        access to embedding weights for in-place normalization.
        """
        import math
        # Access relation embedding weights directly
        phases = self.relation_emb.weight.data
        
        # Shift by pi: phases now in [0, 2*pi) after modulo
        phases = phases + math.pi
        
        # Apply modulo to bring to [0, 2*pi)
        phases = torch.remainder(phases, 2.0 * math.pi)
        
        # Shift back to [-pi, pi]
        phases = phases - math.pi
        
        # Write back (in-place update)
        self.relation_emb.weight.data[:] = phases[:]

    def score_emb(self, h_emb, r_emb, t_emb, score_mode="triple"):
        """Compute RotatE scores: -|| h ⊙ r - t ||
        
        Where r is converted from angles to complex rotations: r = cos(theta) + i*sin(theta)
        
        Args:
            h_emb: [batch, 2*emb_dim] - complex entity embeddings
            r_emb: [batch, emb_dim] - relation angles (phases)
            t_emb: [batch, 2*emb_dim] or [batch, 1+num_neg, 2*emb_dim] - complex entity embeddings
            score_mode: 'triple', 'multi_tails', 'multi_heads'
        """
        n = r_emb.size(0)
        h_emb = self.entity_dropout(h_emb)
        r_emb = self.relation_dropout(r_emb)
        t_emb = self.entity_dropout(t_emb)
        
        # Split entity embeddings into real and imaginary parts
        if score_mode == "triple":
            # h_emb, t_emb: [batch, 2*emb_dim]
            h_re, h_im = h_emb.chunk(2, dim=1)
            t_re, t_im = t_emb.chunk(2, dim=1)
            
            # Convert relation angles to complex numbers on unit circle
            r_re = torch.cos(r_emb)  # [batch, emb_dim]
            r_im = torch.sin(r_emb)  # [batch, emb_dim]
            
            # Compute h ⊙ r (complex Hadamard product)
            hr_re = h_re * r_re - h_im * r_im
            hr_im = h_re * r_im + h_im * r_re
            
            # Compute h ⊙ r - t (complex difference)
            diff_re = hr_re - t_re
            diff_im = hr_im - t_im
            
            # Compute magnitude: sqrt(real^2 + imag^2) using stable method
            # Stack real and imaginary parts and compute norm along new dimension
            diff_stacked = torch.stack([diff_re, diff_im], dim=0)  # [2, batch, emb_dim]
            diff_abs = torch.norm(diff_stacked, dim=0)  # [batch, emb_dim]
            
            # Compute norm and negate
            out = -torch.norm(diff_abs, dim=1, p=self._norm)
            
        elif score_mode == "multi_tails":
            # h_emb: [batch, 2*emb_dim], t_emb: [batch, 1+num_neg, 2*emb_dim]
            h_re, h_im = h_emb.chunk(2, dim=1)  # [batch, emb_dim]
            t_re, t_im = t_emb.chunk(2, dim=2)  # [batch, 1+num_neg, emb_dim]
            
            r_re = torch.cos(r_emb)  # [batch, emb_dim]
            r_im = torch.sin(r_emb)
            
            # Compute h ⊙ r
            hr_re = h_re * r_re - h_im * r_im  # [batch, emb_dim]
            hr_im = h_re * r_im + h_im * r_re
            
            # Expand for broadcasting: [batch, 1, emb_dim]
            hr_re = hr_re.unsqueeze(1)
            hr_im = hr_im.unsqueeze(1)
            
            # Compute pairwise difference
            diff_re = hr_re - t_re  # [batch, 1+num_neg, emb_dim]
            diff_im = hr_im - t_im
            
            # Magnitude using stable method
            diff_stacked = torch.stack([diff_re, diff_im], dim=0)  # [2, batch, 1+num_neg, emb_dim]
            diff_abs = torch.norm(diff_stacked, dim=0)  # [batch, 1+num_neg, emb_dim]
            
            # Norm across embedding dimension
            out = -torch.norm(diff_abs, dim=2, p=self._norm)  # [batch, 1+num_neg]
            
        elif score_mode == "multi_heads":
            # h_emb: [batch, 1+num_neg, 2*emb_dim], t_emb: [batch, 2*emb_dim]
            # For multi_heads, we use the property: || h ⊙ r - t || = || h - conj(r) ⊙ t ||
            # where conj(r) is complex conjugate of r
            h_re, h_im = h_emb.chunk(2, dim=2)  # [batch, 1+num_neg, emb_dim]
            t_re, t_im = t_emb.chunk(2, dim=1)  # [batch, emb_dim]
            
            r_re = torch.cos(r_emb)  # [batch, emb_dim]
            r_im = -torch.sin(r_emb)  # Conjugate: negate imaginary part
            
            # Compute conj(r) ⊙ t
            rt_re = r_re * t_re - r_im * t_im  # [batch, emb_dim]
            rt_im = r_re * t_im + r_im * t_re
            
            # Expand for broadcasting
            rt_re = rt_re.unsqueeze(1)  # [batch, 1, emb_dim]
            rt_im = rt_im.unsqueeze(1)
            
            # Compute pairwise difference
            diff_re = h_re - rt_re  # [batch, 1+num_neg, emb_dim]
            diff_im = h_im - rt_im
            
            # Magnitude using stable method
            diff_stacked = torch.stack([diff_re, diff_im], dim=0)  # [2, batch, 1+num_neg, emb_dim]
            diff_abs = torch.norm(diff_stacked, dim=0)  # [batch, 1+num_neg, emb_dim]
            
            # Norm
            out = -torch.norm(diff_abs, dim=2, p=self._norm)
            
        else:
            raise ValueError(f"Unknown score_mode: {score_mode}")
        
        return out.view(n, -1)

    def _forward_internal(self, heads, relations, tails, score_mode="triple"):
        h = self.entity_emb(heads)
        r = self.relation_emb(relations)
        t = self.entity_emb(tails)
        
        # Normalize phases if enabled (during training)
        if self.training and self._normalize_phases:
            self.normalize_phases()
        
        return self.score_emb(h, r, t, score_mode=score_mode)


import warnings
class ConvE(KGEModel):
    def __init__(self, num_entities, num_relations, embedding_dim, seed_forward, seed_init, filter_size, padding, stride, feature_map_drop, hidden_drop, dropout_entity, dropout_relation, use_inverse, embedding_shape1, init_function):
        super().__init__(num_entities, num_relations, embedding_dim, seed_forward, use_inverse=use_inverse)
        set_seed(seed_init)
        self.entity_emb = nn.Embedding(num_entities, embedding_dim)
        self.entity_bias = nn.Embedding(num_entities, 1)
        self.relation_emb = nn.Embedding(num_relations, embedding_dim)
        self.entity_dropout = nn.Dropout(dropout_entity)
        self.relation_dropout = nn.Dropout(dropout_relation)
        self.feature_map_drop = nn.Dropout2d(feature_map_drop)
        self.hidden_drop = nn.Dropout(hidden_drop)
        self.emb_height = embedding_shape1
        self.emb_width = embedding_dim // embedding_shape1
        self.conv1 = nn.Conv2d(1, 32, (filter_size, filter_size), stride=stride, padding=padding, bias=True)
        self.bn0 = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm1d(embedding_dim)
        conv_out_h = self.emb_height * 2
        conv_out_w = self.emb_width
        conv_out_h = (conv_out_h + 2*padding - filter_size)//stride + 1  # padding=padding, kernel=filter_size, stride=stride
        conv_out_w = (conv_out_w + 2*padding - filter_size)//stride + 1
        self.fc = nn.Linear(32 * conv_out_h * conv_out_w, embedding_dim)
        self.seed_init = seed_init
        self.init_function = get_init_function(init_function)
        self.reset_parameters()
        if not use_inverse:
            warnings.warn("ConvE needs inverse relations (--use_inverse)", UserWarning)

    def reset_parameters(self):
        set_seed(self.seed_init)
        print("Resetting parameters with seed:", self.seed_init)
        self.init_function(self.entity_emb.weight.data)
        self.init_function(self.relation_emb.weight.data)
        self.init_function(self.entity_bias.weight.data) # With other implementations hack, this bias is init as the (d+1)th column of the entity embedding matrix

        # I don't reinitialise the weights and biases of the conv and fc layers
        # It will be initialised based on seed_init with the default pytorch initialisation, as LibKGE do
        print("hash of parameters:", hash(self.entity_emb.weight.data), hash(self.relation_emb.weight.data), hash(self.entity_bias.weight.data))

    def score_emb(self, h_emb, r_emb, t_emb, score_mode="triple", biases=None):

        if score_mode == "multi_heads":
            # ConvE does not support multi_heads mode
            raise NotImplementedError("ConvE does not support 'multi_heads' scoring mode. Only 'triple' and 'multi_tails' are supported.")

        # h_emb, r_emb, t_emb: [batch, emb_dim] or [batch, 1+num_neg, emb_dim]
        # score_mode: 'triple', 'multi_tails', 'multi_heads'
        t_emb = self.entity_dropout(t_emb)
        batch_size = r_emb.size(0)
        # 2D reshape
        h2d = h_emb.view(-1, 1, self.emb_height, self.emb_width)
        r2d = r_emb.view(-1, 1, self.emb_height, self.emb_width)
        x = torch.cat([h2d, r2d], 2)

        x = self.bn0(x)
        x = self.entity_dropout(x)  # like input dropout in the original paper
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        if score_mode == "multi_tails":
            # print(f"t_emb: {t_emb.shape}, x: {x.shape}")
            # t_emb: [batch, 1+num_neg, emb_dim]
            # x: [batch, emb_dim]
            # Compute scores for each (h, r) against all tails in the batch
            out = torch.bmm(t_emb, x.unsqueeze(2)).squeeze(2)  # [batch, 1+num_neg]
            if biases is not None:
                out += biases.squeeze(-1)
        elif score_mode == "triple":
            out = (x * t_emb).sum(dim=1) # [batch]
            if biases is not None:
                out += biases.squeeze(-1)
        else:
            raise ValueError(f"ConvE only supports 'triple' and 'multi_tails' scoring, got {score_mode}")
        return out.view(batch_size, -1)

    def _forward_internal(self, heads, relations, tails, score_mode="triple"):
        h = self.entity_emb(heads)
        r = self.relation_emb(relations)
        t = self.entity_emb(tails)
        biases = self.entity_bias.weight[tails]  # [batch, 1] if score_mode == "triple" else [1+num_neg, 1]
        return self.score_emb(h, r, t, score_mode=score_mode, biases=biases)


class FixedModel(KGEModel):
    def __init__(self, num_entities, num_relations, embedding_dim, seed_forward, seed_init, use_inverse, dropout_entity, dropout_relation, init_function):
        super().__init__(num_entities, num_relations, embedding_dim, seed_forward, use_inverse=use_inverse)
        set_seed(seed_init)
        self.entity_emb = nn.Embedding(num_entities, 1)  # 1D embeddings since we're just using indices
        self.relation_emb = nn.Embedding(num_relations, 1)  # 1D embeddings since we're just using indices
        self.entity_dropout = nn.Dropout(dropout_entity)
        self.relation_dropout = nn.Dropout(dropout_relation)
        self.seed_init = seed_init
        self.init_function = get_init_function(init_function)
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize embeddings to be indices (0 to num_entities-1 for entities, 0 to num_relations-1 for relations)
        with torch.no_grad():
            self.entity_emb.weight.data = torch.arange(self.num_entities, dtype=torch.float32).view(-1, 1).to(self.entity_emb.weight.device)
            self.relation_emb.weight.data = torch.arange(self.num_relations, dtype=torch.float32).view(-1, 1).to(self.relation_emb.weight.device)

    def score_emb(self, h_emb, r_emb, t_emb, score_mode="triple"):
        self.reset_parameters()
        # h_emb, r_emb, t_emb: [batch, 1] or [batch, 1+num_neg, 1]
        # score_mode: 'triple', 'multi_tails', 'multi_heads'
        h_emb = self.entity_dropout(h_emb)
        r_emb = self.relation_dropout(r_emb)
        t_emb = self.entity_dropout(t_emb)
        
        # score = h * num_entities * num_relations + r * num_entities + t
        if score_mode == "triple":
            score = h_emb * self.num_entities * self.num_relations + r_emb * self.num_entities + t_emb
        elif score_mode == "multi_tails":
            # h_emb: [batch, 1], r_emb: [batch, 1], t_emb: [batch, 1+num_neg, 1]
            score = h_emb.unsqueeze(1) * self.num_entities * self.num_relations + r_emb.unsqueeze(1) * self.num_entities + t_emb
        elif score_mode == "multi_heads":
            # t_emb: [batch, 1], r_emb: [batch, 1], h_emb: [batch, 1+num_neg, 1]
            score = h_emb * self.num_entities * self.num_relations + r_emb.unsqueeze(1) * self.num_entities + t_emb.unsqueeze(1)
        else:
            raise ValueError(f"Unknown score_mode: {score_mode}")
            
        return score.squeeze(-1)  # Remove the last dimension to match expected shape

    def _forward_internal(self, heads, relations, tails, score_mode="triple"):
        h = self.entity_emb(heads).squeeze(-1)  # [batch] or [batch, 1+num_neg]
        r = self.relation_emb(relations).squeeze(-1)  # [batch] or [batch, 1+num_neg]
        t = self.entity_emb(tails).squeeze(-1)  # [batch] or [batch, 1+num_neg]
        return self.score_emb(
            h.unsqueeze(-1) if h.dim() == 1 else h.unsqueeze(-1),
            r.unsqueeze(-1) if r.dim() == 1 else r.unsqueeze(-1),
            t.unsqueeze(-1) if t.dim() == 1 else t.unsqueeze(-1),
            score_mode=score_mode
        )

class RGCN(KGEModel):
    """Relational GCN encoder + DistMult scorer (Schlichtkrull et al., 2018)."""


    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int,
        edge_index: torch.LongTensor,
        edge_type: torch.LongTensor,
        seed_forward: int,
        seed_init: int,
        use_inverse: bool,
        dropout_entity: float,
        dropout_relation: float,
        init_function: str,
        num_bases: int = 2,
        use_batched_encoding: bool = True,
        encoder_batch_size: int = 256,  # Batch size for encoding entities in eval mode
        # block_size: int = 2,
    ):
        super().__init__(num_entities, num_relations, embedding_dim, seed_forward, use_inverse=use_inverse)

        set_seed(seed_init)
        import os
        # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "caching_allocator"

        # Lazy import torch_geometric only when RGCN is actually instantiated,
        # so importing kge.models does not require torch_geometric/numba.
        try:
            from torch_geometric.nn import RGCNConv, FastRGCNConv  # type: ignore
            self.RGCNConv = RGCNConv
            self.FastRGCNConv = FastRGCNConv
        except Exception as e:
            raise ImportError(
                "RGCN requires torch-geometric and its dependencies (e.g., numba). "
                "Please install compatible versions or avoid selecting model=RGCN.\n"
                f"Original import error: {e}"
            )

        # Store the graph structure as buffers so they are moved with the model
        self.register_buffer("edge_index", edge_index)
        self.register_buffer("edge_type", edge_type)

        print("edge_index shape:", edge_index.shape, edge_index[:5])
        print("edge_type shape:", edge_type.shape, edge_type[:5])

        set_seed(seed_init)
        self.entity_emb = nn.Embedding(num_entities, embedding_dim)
        self.relation_emb = nn.Embedding(num_relations, embedding_dim)

        self.entity_dropout = nn.Dropout(dropout_entity)
        self.relation_dropout = nn.Dropout(dropout_relation)
        self.hidden_dropout = nn.Dropout(0.2)

        # For meta-model (with old runs)
        # Two-layer R-GCN (emb_dim → emb_dim)
        # Use num_bases parameter to match saved models (default: 2)
        # Old code used num_blocks=embedding_dim//4, but saved models use num_bases=2
        self.conv1 = self.RGCNConv(embedding_dim, embedding_dim, num_relations, num_bases=num_bases)
        self.conv2 = self.RGCNConv(embedding_dim, embedding_dim, num_relations, num_bases=num_bases)


        # For new runs
        # Two-layer R-GCN (emb_dim → emb_dim)
        # self.conv1 = self.RGCNConv(embedding_dim, embedding_dim, num_relations, num_bases=num_bases)
        # self.conv2 = self.RGCNConv(embedding_dim, embedding_dim, num_relations, num_bases=num_bases) 
        # self.conv1 = self.RGCNConv(embedding_dim, embedding_dim, num_relations, num_blocks=embedding_dim//4)
        # self.conv2 = self.RGCNConv(embedding_dim, embedding_dim, num_relations, num_blocks=embedding_dim//4)

        self.seed_init = seed_init
        self.init_function = get_init_function(init_function)
        self.use_batched_encoding = use_batched_encoding
        self.encoder_batch_size = encoder_batch_size
        self.reset_parameters()

    def reset_parameters(self):
        set_seed(self.seed_init)
        print("Resetting parameters with seed:", self.seed_init)
        self.init_function(self.entity_emb.weight.data)
        self.init_function(self.relation_emb.weight.data)
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        print("hash of parameters:", hash(self.entity_emb.weight.data), hash(self.relation_emb.weight.data), hash(self.conv1.weight.data), hash(self.conv2.weight.data))

    def _get_k_hop_subgraph(self, node_idx: torch.Tensor, num_hops: int = 2):
        """Extract k-hop subgraph for given nodes.
        
        Args:
            node_idx: Tensor of node indices [batch_size]
            num_hops: Number of hops (default=2 for 2-layer RGCN)
            
        Returns:
            subset: All nodes in the k-hop subgraph
            sub_edge_index: Edge indices in the subgraph
            mapping: Mapping from original node indices to subgraph indices
            edge_mask: Mask for edges in the subgraph
        """
        from torch_geometric.utils import k_hop_subgraph
        
        subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx=node_idx,
            num_hops=num_hops,
            edge_index=self.edge_index,
            relabel_nodes=True,
            num_nodes=self.entity_emb.num_embeddings,
        )
        
        # Get corresponding edge types
        sub_edge_type = self.edge_type[edge_mask]
        
        return subset, sub_edge_index, mapping, sub_edge_type
    
    def encode_batch(self, batch_entities: torch.Tensor) -> torch.Tensor:
        """Encode a batch of entities using 2-hop subgraph.
        
        Args:
            batch_entities: Tensor of entity indices [batch_size]
            
        Returns:
            Encoded representations for batch_entities [batch_size, emb_dim]
        """
        # Get 2-hop subgraph for the batch
        subset, sub_edge_index, mapping, sub_edge_type = self._get_k_hop_subgraph(
            batch_entities, num_hops=2
        )
        
        # Get embeddings for all nodes in the subgraph
        x = self.entity_emb(subset)  # [num_nodes_in_subgraph, emb_dim]
        
        # Apply 2-layer R-GCN on the subgraph
        x = self.conv1(x, sub_edge_index, sub_edge_type)
        x = F.relu(x)
        x = self.hidden_dropout(x)
        x = self.conv2(x, sub_edge_index, sub_edge_type)
        
        # Return only the embeddings for the requested batch entities
        # mapping contains the indices of batch_entities in the subgraph
        return x[mapping]  # [batch_size, emb_dim]
    
    def encode(self) -> torch.Tensor:
        """Compute entity representations with two R-GCN layers for all entities.
        
        If use_batched_encoding is True and we're not in training mode, this will
        encode entities in batches to save memory.
        """
        if self.use_batched_encoding and not self.training:
            # Encode all entities in batches
            num_entities = self.entity_emb.num_embeddings
            device = self.entity_emb.weight.device
            encoded = torch.zeros(num_entities, self.entity_emb.embedding_dim, device=device)
            
            for start_idx in range(0, num_entities, self.encoder_batch_size):
                end_idx = min(start_idx + self.encoder_batch_size, num_entities)
                batch_entities = torch.arange(start_idx, end_idx, device=device)
                encoded[start_idx:end_idx] = self.encode_batch(batch_entities)
            
            return encoded
        else:
            # Standard full encoding
            x = self.entity_emb.weight
            # x = self.hidden_dropout(x)  # [num_entities, emb_dim]
            x = self.conv1(x, self.edge_index, self.edge_type)
            x = F.relu(x)
            x = self.hidden_dropout(x)
            x = self.conv2(x, self.edge_index, self.edge_type)
            return x  # [num_entities, emb_dim]

    # ------------------------------------------------------------------
    # Scoring (DistMult)
    # ------------------------------------------------------------------
    def score_emb(self, h_emb, r_emb, t_emb, score_mode: str = "triple"):
        n = r_emb.size(0)
        h_emb = self.entity_dropout(h_emb)
        r_emb = self.relation_dropout(r_emb)
        t_emb = self.entity_dropout(t_emb)

        if score_mode == "triple":
            out = (h_emb * r_emb * t_emb).sum(dim=1)
        elif score_mode == "multi_tails":
            left = h_emb * r_emb  # [batch, emb_dim]
            out = torch.bmm(left.unsqueeze(1), t_emb.transpose(1, 2)).squeeze(1)
        elif score_mode == "multi_heads":
            left = t_emb * r_emb  # [batch, emb_dim]
            out = torch.bmm(left.unsqueeze(1), h_emb.transpose(1, 2)).squeeze(1)
        else:
            raise ValueError(f"Unknown score_mode: {score_mode}")
        return out.view(n, -1)

    def _forward_internal(self, heads, relations, tails, score_mode: str = "triple"):
        if self.use_batched_encoding and self.training:
            # TRAINING MODE with batched encoding:
            # Extract unique entities from heads and tails based on score_mode
            if score_mode == "triple":
                # heads and tails are 1D: [batch_size]
                # Example: heads=[5, 10, 20], tails=[10, 100, 200]
                # → batch_entities=[5, 10, 20, 100, 200]
                batch_entities = torch.unique(torch.cat([heads, tails]))
            elif score_mode == "multi_tails":
                # heads: [batch_size], tails: [batch_size, num_entities]
                batch_entities = torch.unique(torch.cat([
                    heads,
                    tails.flatten()
                ]))
            elif score_mode == "multi_heads":
                # heads: [batch_size, num_entities], tails: [batch_size]
                batch_entities = torch.unique(torch.cat([
                    heads.flatten(),
                    tails
                ]))
            else:
                raise ValueError(f"Unknown score_mode: {score_mode}")
            
            # Encode only the batch entities using their 2-hop subgraph
            # Example: batch_entities=[5, 10, 20, 100, 200] → encoded_batch has shape [5, emb_dim]
            encoded_batch = self.encode_batch(batch_entities)
            
            # Create a mapping: global_entity_id → position_in_encoded_batch
            # Example with 1000 total entities and batch_entities=[5, 10, 20, 100, 200]:
            # entity_to_idx will be a tensor of size [1000] filled with -1
            # entity_to_idx[5]=0, entity_to_idx[10]=1, entity_to_idx[20]=2,
            # entity_to_idx[100]=3, entity_to_idx[200]=4, all others=-1
            device = encoded_batch.device
            entity_to_idx = torch.full(
                (self.entity_emb.num_embeddings,), 
                -1, 
                dtype=torch.long, 
                device=device
            )
            entity_to_idx[batch_entities] = torch.arange(
                len(batch_entities), 
                device=device
            )
            
            # Convert global entity IDs to local batch indices
            # Example: heads=[5, 10, 20] → h_idx=[0, 1, 2]
            h_idx = entity_to_idx[heads.flatten()].view_as(heads)
            t_idx = entity_to_idx[tails.flatten()].view_as(tails)
            
            # Get the actual embeddings using local indices
            # Example: h = encoded_batch[[0, 1, 2]] → embeddings for entities 5, 10, 20
            h = encoded_batch[h_idx]
            t = encoded_batch[t_idx]
        else:
            # EVALUATION MODE or standard encoding (use_batched_encoding=False):
            # Encode all entities (in batches if use_batched_encoding=True)
            if self.training:
                # During training without batched encoding, recompute everything
                encoded = self.encode()
            else:
                # During evaluation, cache the full encoding
                if not hasattr(self, '_cached_encoded'):
                    self._cached_encoded = self.encode()
                encoded = self._cached_encoded
                
            # Simply index into the full encoded tensor
            h = encoded[heads]
            t = encoded[tails]
        
        r = self.relation_emb(relations)
        return self.score_emb(h, r, t, score_mode=score_mode)
        
    def train(self, mode: bool = True):
        """Override train() to clear the cached encoded representations when switching to train mode."""
        super().train(mode)
        if mode and hasattr(self, '_cached_encoded'):
            del self._cached_encoded


class Transformer(KGEModel):
    """Transformer-based KGE scorer inspired by LibKGE's implementation of no-context transformer of Hitter paper.
    https://github.com/uma-pi1/kge/blob/master/kge/model/transformer.py

    Conventions aligned with this codebase:
    - parameter loading via `init_function` and `seed_init`
    - dropout on entity/relation inputs
    - `score_emb` supports 'triple' and 'multi_tails'
    - direct 'multi_heads' is not supported; rely on base class + use_inverse=True
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int,
        seed_forward: int,
        seed_init: int,
        use_inverse: bool,
        dropout_entity: float,
        dropout_relation: float,
        init_function: str,
        encoder_nhead: int = 8,  # like LibkGE config
        encoder_dim_feedforward: int = 1280,  # like LibkGE config
        encoder_num_layers: int = 3,  # like LibkGE config
        encoder_activation: str = "relu",  # like LibkGE config
        encoder_dropout: float = 0.1,  # like LibkGE config
    ):
        super().__init__(num_entities, num_relations, embedding_dim, seed_forward, use_inverse=use_inverse)

        set_seed(seed_init)
        # Embeddings
        self.entity_emb = nn.Embedding(num_entities, embedding_dim)
        self.relation_emb = nn.Embedding(num_relations, embedding_dim)
        # Learnable CLS + type embeddings
        self.cls_emb = nn.Parameter(torch.zeros(embedding_dim))
        self.sub_type_emb = nn.Parameter(torch.zeros(embedding_dim))
        self.rel_type_emb = nn.Parameter(torch.zeros(embedding_dim))

        # Dropouts
        self.entity_dropout = nn.Dropout(dropout_entity)
        self.relation_dropout = nn.Dropout(dropout_relation)

        # Transformer encoder (sequence length 3; [S, N, E] expected)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=encoder_nhead,
            dim_feedforward=encoder_dim_feedforward,
            dropout=encoder_dropout,
            activation=encoder_activation,
            batch_first=False,
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=encoder_num_layers)

        self.seed_init = seed_init
        self.init_function = get_init_function(init_function)
        self.reset_parameters()

    def reset_parameters(self):
        set_seed(self.seed_init)
        self.init_function(self.entity_emb.weight.data)
        self.init_function(self.relation_emb.weight.data)


        # Initialization for transformer layers, CLS embedding and type embeddings like LibkGE config
        torch.nn.init.normal_(self.cls_emb.data, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.sub_type_emb.data, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.rel_type_emb.data, mean=0.0, std=0.02)

        for layer in self.encoder.layers:
            torch.nn.init.normal_(layer.linear1.weight.data)
            torch.nn.init.normal_(layer.linear2.weight.data)
            torch.nn.init.normal_(layer.self_attn.out_proj.weight.data)
            if getattr(layer.self_attn, "_qkv_same_embed_dim", False):
                torch.nn.init.normal_(layer.self_attn.in_proj_weight.data)
            else:
                torch.nn.init.normal_(layer.self_attn.q_proj_weight.data)
                torch.nn.init.normal_(layer.self_attn.k_proj_weight.data)
                torch.nn.init.normal_(layer.self_attn.v_proj_weight.data)
        

    def score_emb(self, h_emb, r_emb, t_emb, score_mode: str = "triple"):
        # Important: if multi_heads, do NOT build the [CLS, h, r] sequence here, as h_emb
        # may have shape [batch, 1+num_neg, emb] and will break stacking. We don't implement
        # direct multi_heads; require use_inverse=True to route to multi_tails.
        if score_mode == "multi_heads":
            raise NotImplementedError(
                "Transformer does not support 'multi_heads' directly; set use_inverse=True to route to multi_tails with inverse relations."
            )

        n = r_emb.size(0)
        h_emb = self.entity_dropout(h_emb)
        r_emb = self.relation_dropout(r_emb)
        t_emb = self.entity_dropout(t_emb)

        seq = torch.stack(
            (
                self.cls_emb.unsqueeze(0).repeat((n, 1)),
                h_emb + self.sub_type_emb.unsqueeze(0),
                r_emb + self.rel_type_emb.unsqueeze(0),
            ),
            dim=0,
        )  # [3, n, d]

        out = self.encoder(seq)
        cls_transformed = out[0, :, :]

        if score_mode == "triple":
            scores = (cls_transformed * t_emb).sum(dim=1)
        elif score_mode == "multi_tails":
            scores = torch.bmm(t_emb, cls_transformed.unsqueeze(2)).squeeze(2)
        else:
            raise ValueError(f"Unknown score_mode: {score_mode}")
        return scores.view(n, -1)

    def _forward_internal(self, heads, relations, tails, score_mode: str = "triple"):
        h = self.entity_emb(heads)
        r = self.relation_emb(relations)
        t = self.entity_emb(tails)
        return self.score_emb(h, r, t, score_mode=score_mode)