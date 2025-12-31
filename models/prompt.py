from transformers.models.bert.modeling_bert import BertModel, BertPreTrainedModel, BertOnlyMLMHead
from transformers.modeling_outputs import (
    MaskedLMOutput
)
from transformers.activations import ACT2FN
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn as nn
import torch
from transformers import AutoTokenizer
import os
from .loss import multilabel_categorical_crossentropy
from .graph import GraphEncoder
from .attention import CrossAttention
import math
import torch.nn.functional as F
from sklearn.metrics import f1_score
import json


class GraphEmbedding(nn.Module):
    def __init__(self, config, embedding, new_embedding, graph_type='GAT', layer=1, path_list=None, data_path=None):
        super(GraphEmbedding, self).__init__()
        self.graph_type = graph_type
        padding_idx = config.pad_token_id
        self.num_class = config.num_labels
        if self.graph_type != '':
            self.graph = GraphEncoder(config, graph_type, layer, path_list=path_list, data_path=data_path)
        self.padding_idx = padding_idx
        self.original_embedding = embedding
        new_embedding = torch.cat(
            [torch.zeros(1, new_embedding.size(-1), device=new_embedding.device, dtype=new_embedding.dtype),
             new_embedding], dim=0)
        self.new_embedding = nn.Embedding.from_pretrained(new_embedding, True, 0)
        self.size = self.original_embedding.num_embeddings + self.new_embedding.num_embeddings - 1
        self.depth = (self.new_embedding.num_embeddings - 2 - self.num_class)

    @property
    def weight(self):
        def foo():
            # label prompt MASK
            edge_features = self.new_embedding.weight[1:, :]
            if self.graph_type != '':
                # label prompt
                edge_features = edge_features[:-1, :]
                edge_features = self.graph(edge_features, self.original_embedding)
                edge_features = torch.cat(
                    [edge_features, self.new_embedding.weight[-1:, :]], dim=0)
            return torch.cat([self.original_embedding.weight, edge_features], dim=0)

        return foo

    @property
    def raw_weight(self):
        def foo():
            return torch.cat([self.original_embedding.weight, self.new_embedding.weight[1:, :]], dim=0)

        return foo

    def forward(self, x):
        x = F.embedding(x, self.weight(), self.padding_idx)

        return x


class OutputEmbedding(nn.Module):
    def __init__(self, bias):
        super(OutputEmbedding, self).__init__()
        self.weight = None
        self.bias = bias

    def forward(self, x):
        return F.linear(x, self.weight(), self.bias)


class Prompt(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config, graph_type='GAT', layer=1, path_list=None, data_path=None, depth2label=None, use_label_description=False, **kwargs):
        super().__init__(config)    

        self.bert = BertModel(config, add_pooling_layer=False)
        self.tokenizer = AutoTokenizer.from_pretrained(self.name_or_path)
        self.cls = BertOnlyMLMHead(config)
        self.num_labels = config.num_labels
        self.multiclass_bias = nn.Parameter(torch.zeros(self.num_labels, dtype=torch.float32))
        bound = 1 / math.sqrt(768)
        nn.init.uniform_(self.multiclass_bias, -bound, bound)
        self.data_path = data_path
        self.graph_type = graph_type
        self.vocab_size = self.tokenizer.vocab_size
        self.path_list = path_list
        self.depth2label = depth2label
        self.layer = layer
        self.use_label_description = use_label_description
        self.init_weights()

        if self.data_path.split('/')[-1] == 'nyt':
            label_dict = torch.load(os.path.join(self.data_path, 'label_dict.pt'))
        else:
            label_dict = torch.load(os.path.join(self.data_path, 'value_dict.pt'))
        label_dict = {v: i for i, v in label_dict.items()}
        label_dict['Root'] = -1
        taxnomy = []

        if self.data_path.split('/')[-1] == 'nyt':
            with open(os.path.join(self.data_path, 'nyt.taxonomy'), 'r') as f:
                for line in f.readlines():
                    line = line.strip().split('\t')
                    taxnomy.append(line)
        if self.data_path.split('/')[-1] == 'rcv1':
            with open(os.path.join(self.data_path, 'rcv1.taxonomy'), 'r') as f:
                for line in f.readlines():
                    line = line.strip().split('\t')
                    taxnomy.append(line)
        if self.data_path.split('/')[-1] == 'WebOfScience':
            with open(os.path.join(self.data_path, 'wos.taxonomy'), 'r') as f:
                for line in f.readlines():
                    line = line.strip().split('\t')
                    taxnomy.append(line)

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def _gram_schmidt(self, vectors):
        # Gram-Schmidt orthogonalization of a set of vectors
        basis = []
        for v in vectors:
            w = v - sum(torch.dot(v, b) / torch.dot(b, b) * b for b in basis)
            if torch.linalg.norm(w) > 1e-8:
                basis.append(w / torch.linalg.norm(w))
        return torch.stack(basis)

    def semantically_anchored_gof_initialization(self, E_initial, depth2label, value2slot):
        print("Performing Semantically-Anchored GOF Initialization...")
        E_structured = E_initial.clone()
        
        # 1. Orthogonalization of head labels at the top level
        top_level_indices = depth2label.get(0, [])
        if len(top_level_indices) > 0:
            top_vectors_initial = E_structured[top_level_indices]
            top_vectors_ortho = self._gram_schmidt(top_vectors_initial)
            # Replace the orthogonalized vector back in its original position, while maintaining its original length.
            original_norms = torch.linalg.norm(top_vectors_initial, dim=1, keepdim=True)
            E_structured[top_level_indices] = top_vectors_ortho * original_norms

        # 2. Process child nodes layer by layer from top to bottom, and orthogonalize sibling residuals.
        parent_to_children = {}
        for child, parent in value2slot.items():
            if parent != -1:
                if parent not in parent_to_children: parent_to_children[parent] = []
                parent_to_children[parent].append(child)

        max_depth = max(depth2label.keys())
        for d in range(max_depth + 1):
            for parent_idx in depth2label.get(d, []):
                children_indices = parent_to_children.get(parent_idx, [])
                if len(children_indices) > 1:
                    E_p = E_structured[parent_idx]
                    
                    # Calculate the residual vector of all siblings
                    residuals = []
                    for child_idx in children_indices:
                        E_c = E_structured[child_idx]
                        proj = (torch.dot(E_c, E_p) / torch.dot(E_p, E_p)) * E_p
                        residuals.append(E_c - proj)
                    
                    # Orthogonalize the residual vector
                    if len(residuals) > 0:
                        ortho_residuals = self._gram_schmidt(residuals)
                        original_res_norms = torch.linalg.norm(torch.stack(residuals), dim=1, keepdim=True)
                        
                        # Reconstructing subvectors
                        for i, child_idx in enumerate(children_indices):
                             E_c_initial = E_structured[child_idx]
                             proj = (torch.dot(E_c_initial, E_p) / torch.dot(E_p, E_p)) * E_p
                             # Using orthogonalized residual directions and the original residual length
                             if i < len(ortho_residuals):
                                E_structured[child_idx] = proj + ortho_residuals[i] * original_res_norms[i]
        
        return E_structured

    def init_embedding(self):
        depth = len(self.depth2label)
        tokenizer = AutoTokenizer.from_pretrained(self.name_or_path)
        label_emb_list = []

        if self.use_label_description:
            print("Initializing label embeddings using detailed descriptions...")
            descriptions_path = os.path.join(self.data_path, 'label_descriptions.json')
            with open(descriptions_path, 'r', encoding='utf-8') as f:
                label_descriptions = json.load(f)
            label_descriptions = {int(k): v for k, v in label_descriptions.items()}
            max_description_length = 512 # Limit the maximum length of the description
            label_tokens = {i: tokenizer.encode(v, truncation=True, max_length=max_description_length) 
                            for i, v in label_descriptions.items()}
            input_embeds = self.get_input_embeddings()
            for i in range(self.num_labels):
                token_ids = label_tokens.get(i)
                label_emb_list.append(input_embeds.weight.index_select(0, torch.tensor(token_ids, device=self.device)).mean(dim=0))
        else:
            print("Initializing label embeddings using original label names...")
            label_dict = torch.load(os.path.join(self.data_path, 'value_dict.pt')) # 0: CS
            label_dict = {i: tokenizer.encode(v) for i, v in label_dict.items()} # {0: [5236, 6191], 1: [2974, 5236, 6191], 2: [2974, 3241, 5236, 6191]}
            input_embeds = self.get_input_embeddings()
            for i in range(len(label_dict)):
                label_emb_list.append(
                    input_embeds.weight.index_select(0, torch.tensor(label_dict[i], device=self.device)).mean(dim=0))

        # 1. Obtain the initial, semantically rich embeddings.
        E_initial = torch.stack(label_emb_list) 
        # E_initial = torch.randn_like(E_initial)
        # 2. By semantically anchoring it with GOF initialization, the ideal geometric structure can be obtained.
        value2slot = {child: parent for child, parent in self.path_list 
                      if parent != -1 and child < self.num_labels and parent < self.num_labels}
        for i in range(self.num_labels):
            if i not in value2slot: value2slot[i] = -1 # Ensure all nodes are present
        ideal_label_embeddings = self.semantically_anchored_gof_initialization(E_initial, self.depth2label, value2slot)
        ## 3. The ideal embedding is stored as a persistent buffer in the model so that it can be used as a target during training.
        # self.register_buffer('ideal_label_embeddings', ideal_label_embeddings.detach())
        # Convert ideal_label_embeddings into trainable parameters.
        self.ideal_label_embeddings = nn.Parameter(ideal_label_embeddings)


        prefix = input_embeds(torch.tensor([tokenizer.mask_token_id],
                                           device=self.device, dtype=torch.long)) # MASK Embedded Representation
        # prompt
        prompt_embedding = nn.Embedding(depth + 1,
                                        input_embeds.weight.size(1), 0) # 
        self._init_weights(prompt_embedding)

        label_emb = torch.cat(
            [ideal_label_embeddings, prompt_embedding.weight[1:, :], prefix], dim=0)


        embedding = GraphEmbedding(self.config, input_embeds, label_emb, self.graph_type,
                                   path_list=self.path_list, layer=self.layer, data_path=self.data_path)
        self.set_input_embeddings(embedding)
        output_embeddings = OutputEmbedding(self.get_output_embeddings().bias)
        self.set_output_embeddings(output_embeddings)
        output_embeddings.weight = embedding.raw_weight
        self.vocab_size = output_embeddings.bias.size(0)
        output_embeddings.bias.data = nn.functional.pad(
            output_embeddings.bias.data,
            (
                0,
                embedding.size - output_embeddings.bias.shape[0],
            ),
            "constant",
            0,
        )

    def get_layer_features(self, layer, prompt_feature=None):
        labels = torch.tensor(self.depth2label[layer], device=self.device) + 1
        label_features = self.get_input_embeddings().new_embedding(labels)
        label_features = self.transform(label_features)
        label_features = torch.dropout(F.relu(label_features), train=self.training, p=self.config.hidden_dropout_prob)
        return label_features

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        multiclass_pos = input_ids == (self.get_input_embeddings().size - 1)
        single_labels = input_ids.masked_fill(multiclass_pos | (input_ids == self.config.pad_token_id), -100)
        if self.training:
            enable_mask = input_ids < self.tokenizer.vocab_size
            random_mask = torch.rand(input_ids.shape, device=input_ids.device) * attention_mask * enable_mask
            input_ids = input_ids.masked_fill(random_mask > 0.865, self.tokenizer.mask_token_id)
            random_ids = torch.randint_like(input_ids, 104, self.vocab_size)
            mlm_mask = random_mask > 0.985
            input_ids = input_ids * mlm_mask.logical_not() + random_ids * mlm_mask
            mlm_mask = random_mask < 0.85
            single_labels = single_labels.masked_fill(mlm_mask, -100)

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        ) # nan

        sequence_output = outputs[0] # last_states [16,512,768]
        prediction_scores = self.cls(sequence_output) # [16,512,30630]

        masked_lm_loss = None
        total_loss = None
        multiclass_loss = None

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, prediction_scores.size(-1)),
                                      single_labels.view(-1))
            multiclass_logits = prediction_scores.masked_select(
                multiclass_pos.unsqueeze(-1).expand(-1, -1, prediction_scores.size(-1))).view(-1,
                                                                                              prediction_scores.size(
                                                                                                  -1))
            multiclass_logits = multiclass_logits[:,
                                self.vocab_size:self.vocab_size + self.num_labels] + self.multiclass_bias
            multiclass_loss = multilabel_categorical_crossentropy(labels.view(-1, self.num_labels), multiclass_logits)
            total_loss = masked_lm_loss + multiclass_loss

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
        ret = MaskedLMOutput(
            loss=total_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        return ret, masked_lm_loss, multiclass_loss

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]

        assert self.config.pad_token_id is not None, "The PAD token should be defined for generation"
        attention_mask = torch.cat([attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], dim=-1)
        dummy_token = torch.full(
            (effective_batch_size, 1), self.config.pad_token_id, dtype=torch.long, device=input_ids.device
        )
        input_ids = torch.cat([input_ids, dummy_token], dim=1)

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    @torch.no_grad()
    def generate(self, input_ids, depth2label, **kwargs):
        attention_mask = input_ids != self.config.pad_token_id
        outputs, masked_lm_loss, multiclass_loss = self(input_ids, attention_mask)
        multiclass_pos = input_ids == (self.get_input_embeddings().size - 1)
        prediction_scores = outputs['logits']
        prediction_scores = prediction_scores.masked_select(
            multiclass_pos.unsqueeze(-1).expand(-1, -1, prediction_scores.size(-1))).view(-1,
                                                                                          prediction_scores.size(
                                                                                              -1))
        prediction_scores = prediction_scores[:,
                            self.vocab_size:self.vocab_size + self.num_labels] + self.multiclass_bias
        prediction_scores = prediction_scores.view(-1, len(depth2label), prediction_scores.size(-1))
        predict_labels = []
        for scores in prediction_scores:
            predict_labels.append([])
            for i, score in enumerate(scores):
                for l in depth2label[i]:
                    if score[l] > 0:
                        predict_labels[-1].append(l)
        return predict_labels, prediction_scores