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


def quantize_boxes(boxes, coord_max=COORD_MAX):
    """Normalized xyxy [B,4] -> integer coordinate tokens [B,4]."""
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("Expected normalized boxes with shape [B,4]")
    return torch.round(boxes.clamp(0, 1) * coord_max).long()


def teacher_forcing_sequences(boxes, coord_max=COORD_MAX):
    """Decoder inputs [BOS,x1,y1,x2,y2], target [x1,y1,x2,y2,EOS]."""
    coords = quantize_boxes(boxes, coord_max)
    bos = coords.new_full((coords.shape[0], 1), coord_max + 1)
    eos = coords.new_full((coords.shape[0], 1), coord_max + 2)
    return torch.cat([bos, coords], dim=1), torch.cat([coords, eos], dim=1)


def tokens_to_boxes(tokens, coord_max=COORD_MAX):
    """Coordinate tokens -> normalized boxes; keep invalid geometry visible."""
    if tokens.ndim != 2 or tokens.shape[1] != 4:
        raise ValueError("Expected [B,4] coordinate tokens")
    return tokens.float() / coord_max


class GenerativeGrounder(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch16",
                 hidden=256, layers=2, heads=4, dropout=0.1, coord_max=COORD_MAX):
        super().__init__()
        self.model_name = model_name
        self.hidden = hidden
        self.layers = layers
        self.heads = heads
        self.dropout = dropout
        if coord_max not in (100, 224, 1000):
            raise ValueError("coord_max must be one of 100, 224, 1000")
        self.coord_max = coord_max
        self.bos_id = coord_max + 1
        self.eos_id = coord_max + 2
        self.vocab_size = coord_max + 3
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.clip = CLIPModel.from_pretrained(model_name)
        self.clip.requires_grad_(False)
        self.clip.eval()
        projection_dim = self.clip.config.projection_dim
        self.image_projection = nn.Linear(projection_dim, hidden)
        self.text_projection = nn.Linear(projection_dim, hidden)
        self.memory_type = nn.Parameter(torch.zeros(1, 2, hidden))
        self.token_embedding = nn.Embedding(self.vocab_size, hidden)
        self.position_embedding = nn.Embedding(OUTPUT_LENGTH, hidden)
        block = nn.TransformerDecoderLayer(
            d_model=hidden, nhead=heads, dim_feedforward=hidden*4,
            dropout=dropout, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(block, num_layers=layers)
        self.output_norm = nn.LayerNorm(hidden)
        self.lm_head = nn.Linear(hidden, self.vocab_size)

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
        decoder_inputs, targets = teacher_forcing_sequences(target_boxes, self.coord_max)
        memory = self.encode(images, texts)
        logits = self.decode(decoder_inputs.to(memory.device), memory)
        loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.to(memory.device).reshape(-1))
        with torch.no_grad():
            predictions = logits.argmax(-1)
            exact = predictions == targets.to(memory.device)
            token_acc = exact.float().mean()
            coord_acc = exact[:, :4].float().mean()
            eos_acc = exact[:, 4].float().mean()
            # Teacher-forced argmax of coordinate tokens (not autoregressive inference).
            tf_boxes = tokens_to_boxes(predictions[:, :4].clamp(max=self.coord_max), self.coord_max)
        return {"loss": loss, "token_accuracy": token_acc,
                "coordinate_token_accuracy": coord_acc,
                "eos_token_accuracy": eos_acc,
                "teacher_forced_boxes": tf_boxes, "logits": logits}

    @torch.no_grad()
    def generate(self, images, texts, constrain_box=False):
        """Greedy autoregressive decoding: 4 coordinate positions then EOS.

        Coordinates constrained to 0..1000 at each coordinate position.
        EOS is decoded at position 5 to measure correct sequence termination.
        """
        memory = self.encode(images, texts)
        batch = memory.shape[0]
        decoder_ids = torch.full((batch, 1), self.bos_id, device=memory.device, dtype=torch.long)
        chosen = []
        for step in range(OUTPUT_LENGTH):
            next_logits = self.decode(decoder_ids, memory)[:, -1]
            if step < 4:
                coord_logits = next_logits[:, :self.coord_max+1]
                if constrain_box and step in (2, 3):
                    # x2 must exceed x1; y2 must exceed y1. This changes
                    # decoding only; report constraint as an explicit ablation.
                    previous = chosen[0] if step == 2 else chosen[1]
                    values = torch.arange(self.coord_max+1, device=coord_logits.device)
                    coord_logits = coord_logits.masked_fill(
                        values[None, :] <= previous[:, None], float("-inf"))
                token = coord_logits.argmax(-1)
            else:
                token = next_logits.argmax(-1)
            chosen.append(token)
            if step < OUTPUT_LENGTH-1:
                decoder_ids = torch.cat([decoder_ids, token[:, None]], dim=-1)
        sequence = torch.stack(chosen, dim=1)
        return {
            "boxes": tokens_to_boxes(sequence[:, :4], self.coord_max),
            "tokens": sequence,
            "eos_correct": sequence[:, 4] == self.eos_id,
        }
