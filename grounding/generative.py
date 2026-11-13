"""V7 educational generative grounding: coordinate tokens via autoregressive CE.

This is a *small frozen-CLIP + Transformer decoder experiment*, NOT the Qwen3-VL
implementation or a claim to reproduce its training pipeline. Image patches and
a pooled text representation form encoder memory; the decoder predicts
[x1, y1, x2, y2, EOS] as discrete tokens via teacher forcing.
"""
import torch
from torch import nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

COORD_MAX = 1000
BOS = COORD_MAX + 1
EOS = COORD_MAX + 2
VOCAB_SIZE = COORD_MAX + 3
OUTPUT_LENGTH = 5


def quantize_boxes(boxes):
    """Normalized xyxy [B,4] -> integer coordinate tokens [B,4]."""
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("Expected normalized boxes with shape [B,4]")
    return torch.round(boxes.clamp(0, 1) * COORD_MAX).long()


def teacher_forcing_sequences(boxes):
    """Decoder inputs [BOS,x1,y1,x2,y2], target [x1,y1,x2,y2,EOS]."""
    coords = quantize_boxes(boxes)
    bos = coords.new_full((coords.shape[0], 1), BOS)
    eos = coords.new_full((coords.shape[0], 1), EOS)
    return torch.cat([bos, coords], dim=1), torch.cat([coords, eos], dim=1)


def tokens_to_boxes(tokens):
    """Coordinate tokens -> normalized boxes; keep invalid geometry visible."""
    if tokens.ndim != 2 or tokens.shape[1] != 4:
        raise ValueError("Expected [B,4] coordinate tokens")
    return tokens.float() / COORD_MAX


class GenerativeGrounder(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch16",
                 hidden=256, layers=2, heads=4, dropout=0.1):
        super().__init__()
        self.model_name = model_name
        self.hidden = hidden
        self.layers = layers
        self.heads = heads
        self.dropout = dropout
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.clip = CLIPModel.from_pretrained(model_name)
        self.clip.requires_grad_(False)
        self.clip.eval()
        projection_dim = self.clip.config.projection_dim
        self.image_projection = nn.Linear(projection_dim, hidden)
        self.text_projection = nn.Linear(projection_dim, hidden)
        self.memory_type = nn.Parameter(torch.zeros(1, 2, hidden))
        self.token_embedding = nn.Embedding(VOCAB_SIZE, hidden)
        self.position_embedding = nn.Embedding(OUTPUT_LENGTH, hidden)
        block = nn.TransformerDecoderLayer(
            d_model=hidden, nhead=heads, dim_feedforward=hidden*4,
            dropout=dropout, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(block, num_layers=layers)
        self.output_norm = nn.LayerNorm(hidden)
        self.lm_head = nn.Linear(hidden, VOCAB_SIZE)

    def train(self, mode=True):
        super().train(mode)
        self.clip.eval()
        return self

    def encode(self, images, texts):
        """CLIP patch tokens + pooled text token, shape [B,N+1,H]."""
        batch = self.processor(images=images, text=texts, padding=True,
                               truncation=True, return_tensors="pt")
        device = self.token_embedding.weight.device
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            visual = self.clip.vision_model(
                pixel_values=batch["pixel_values"]).last_hidden_state[:, 1:]
            visual = self.clip.visual_projection(visual)
            language = self.clip.text_model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]).pooler_output
            language = self.clip.text_projection(language)
        img_tokens = self.image_projection(visual) + self.memory_type[:, :1]
        text_token = self.text_projection(language).unsqueeze(1) + self.memory_type[:, 1:2]
        return torch.cat([img_tokens, text_token], dim=1)

    def decode(self, decoder_input_ids, memory):
        length = decoder_input_ids.shape[1]
        if not 1 <= length <= OUTPUT_LENGTH:
            raise ValueError("Decoder length must be in [1,5]")
        positions = torch.arange(length, device=decoder_input_ids.device)
        decoder_input = self.token_embedding(decoder_input_ids) + self.position_embedding(positions)[None]
        causal = torch.triu(torch.ones(length, length, device=decoder_input_ids.device,
                                       dtype=torch.bool), diagonal=1)
        hidden = self.decoder(tgt=decoder_input, memory=memory, tgt_mask=causal)
        return self.lm_head(self.output_norm(hidden))

    def forward(self, images, texts, target_boxes):
        """Teacher forcing. Cross-entropy covers all four coordinates and EOS."""
        decoder_inputs, targets = teacher_forcing_sequences(target_boxes)
        memory = self.encode(images, texts)
        logits = self.decode(decoder_inputs.to(memory.device), memory)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), targets.to(memory.device).reshape(-1))
        with torch.no_grad():
            token_acc = (logits.argmax(-1) == targets.to(memory.device)).float().mean()
        return {"loss": loss, "token_accuracy": token_acc, "logits": logits}

    @torch.no_grad()
    def generate(self, images, texts):
        """Greedy autoregressive decoding: 4 coordinate positions then EOS.

        Coordinates constrained to 0..1000 at each coordinate position.
        EOS is decoded at position 5 to measure correct sequence termination.
        """
        memory = self.encode(images, texts)
        batch = memory.shape[0]
        decoder_ids = torch.full((batch, 1), BOS, device=memory.device, dtype=torch.long)
        chosen = []
        for step in range(OUTPUT_LENGTH):
            next_logits = self.decode(decoder_ids, memory)[:, -1]
            if step < 4:
                token = next_logits[:, :COORD_MAX+1].argmax(-1)
            else:
                token = next_logits.argmax(-1)
            chosen.append(token)
            if step < OUTPUT_LENGTH-1:
                decoder_ids = torch.cat([decoder_ids, token[:, None]], dim=-1)
        sequence = torch.stack(chosen, dim=1)
        return {
            "boxes": tokens_to_boxes(sequence[:, :4]),
            "tokens": sequence,
            "eos_correct": sequence[:, 4] == EOS,
        }
