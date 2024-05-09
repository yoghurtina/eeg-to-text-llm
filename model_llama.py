import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
model_id="llama2-hf"
print("hello")

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,  # Or adjust as needed
    torch_dtype=torch.float16
)

# Class for mapping additional encoders to LLaMA
class LlamaTranslator(nn.Module):
    def __init__(self, pretrained_model_dir=model_id, in_feature=840, decoder_embedding_size=768, additional_encoder_nhead=8, additional_encoder_dim_feedforward=2048):
        super(LlamaTranslator, self).__init__()
        
        # Load LLaMA tokenizer and model from a pre-trained directory
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.pretrained = AutoModelForCausalLM.from_pretrained(model_id, device_map='cpu', torch_dtype=torch.float16, quantization_config=quantization_config)  
        # Additional Transformer encoder
        self.additional_encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_feature, nhead=additional_encoder_nhead, dim_feedforward=additional_encoder_dim_feedforward, batch_first=True
        )
        self.additional_encoder = nn.TransformerEncoder(self.additional_encoder_layer, num_layers=6)
        
        # Fully connected layer to match the decoder embedding size
        self.fc1 = nn.Linear(in_feature, decoder_embedding_size)
    
    def addin_forward(self, input_embeddings_batch, input_masks_invert):
        encoded_embedding = self.additional_encoder(
            input_embeddings_batch, src_key_padding_mask=input_masks_invert
        )
        encoded_embedding = F.relu(self.fc1(encoded_embedding))
        return encoded_embedding
    
    def forward(self, input_embeddings_batch, input_masks_batch, input_masks_invert, target_ids_batch_converted):
        # Get encoded embeddings from additional encoder
        encoded_embedding = self.addin_forward(input_embeddings_batch, input_masks_invert)
        
        # Convert embeddings to LLaMA-compatible tokens
        inputs_embeds = encoded_embedding

        # Pass through pre-trained model
        outputs = self.pretrained(
            inputs_embeds=inputs_embeds, attention_mask=input_masks_batch, labels=target_ids_batch_converted, return_dict=True
        )
        return outputs.loss, outputs.logits
    
    @torch.no_grad()
    def generate(self, input_embeddings_batch, input_masks_batch, input_masks_invert, max_length=50):
        encoded_embedding = self.addin_forward(input_embeddings_batch, input_masks_invert)
        generated_ids = self.pretrained.generate(
            inputs_embeds=encoded_embedding,
            attention_mask=input_masks_batch[:, :encoded_embedding.shape[1]],  # Ensure correct mask alignment
            max_length=max_length
        )
        generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return generated_text


# Additional helper classes
class Pooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]  # Pooling from the first token
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Generate positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class EEG2LlamaMapping(nn.Module):
    def __init__(self, in_feature=840, hidden_size=512, out_feature=768):
        super(EEG2LlamaMapping, self).__init__()
        self.fc1 = nn.Linear(in_feature, hidden_size)
        self.fc2 = nn.Linear(hidden_size, out_feature)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        return out
