import torch
import torch.nn as nn
from torch.nn import functional as nnf
from torch.utils.data import Dataset, DataLoader
from enum import Enum
from transformers import GPT2Tokenizer, GPT2LMHeadModel, get_linear_schedule_with_warmup
from torchvision import transforms
from torch.optim import AdamW
from tqdm import tqdm
import os
import sys
import argparse
import json
from typing import Tuple, Optional, Union, List
import ruclip
from PIL import Image
from nltk.translate.bleu_score import corpus_bleu
from nltk import word_tokenize


class MappingType(Enum):
    MLP = 'mlp'
    Transformer = 'transformer'


class MemeDataset(Dataset):

    def __init__(self, prefix_length: int, data_path: str, images_path: str, clip_model: str, tokenizer,
                 normalize_prefix=False,  max_len: int = 100, device: str = 'cpu', mode: str = 'train', max_n: int = 15000):

        self.images_path = images_path
        self.prefix_length = prefix_length
        self.normalize_prefix = normalize_prefix
        clip, processor = ruclip.load(clip_model, device=device)
        self.clip_processor = ruclip.Predictor(clip, processor, device=device, quiet=True)
        self.captions = []
        self.image_ids = []
        # загружаем id картинок и их captions
        with open(data_path, 'r', encoding='utf-8') as f:
            i = 0
            for line in f:
                if i == 0:
                    i += 1
                    continue
                i += 1
                l = line.strip().split('\t')
                if len(l) != 2:
                    continue
                uid, caption = l
                if not os.path.exists(os.path.join(images_path, f'{uid}.jpg')):
                    continue
                self.captions.append(caption)
                self.image_ids.append(uid)
                if max_n and i >= max_n:
                    break


        # переводим captions в gpt-токены
        self.captions_tokens = []
        max_seq_len = 0
        for caption in self.captions:
            self.captions_tokens.append(torch.tensor(tokenizer.encode(str(caption)), dtype=torch.int64))
            max_seq_len = max(max_seq_len, self.captions_tokens[-1].shape[0])
        # максимальная длина последовательности, которая будет подаваться на вход
        self.max_seq_len = max_len

        if mode == 'train':
            self.transform = transforms.Compose([transforms.Resize(256),
                                              transforms.RandomCrop(224),
                                              transforms.RandomHorizontalFlip()])
        else:
            self.transform = transforms.Compose([transforms.Resize((224, 224))])

    def __len__(self) -> int:
        return len(self.captions_tokens)

    def pad_tokens(self, item: int):
        tokens = self.captions_tokens[item]
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
            self.captions_tokens[item] = tokens
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
            self.captions_tokens[item] = tokens
        mask = tokens.ge(0)  # mask is zero where we out of sequence
        tokens[~mask] = 0
        mask = mask.float()
        mask = torch.cat((torch.ones(self.prefix_length), mask), dim=0)  # adding prefix mask
        return tokens, mask

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        tokens, mask = self.pad_tokens(item)
        image_id = self.image_ids[item]
        path = os.path.join(self.images_path, f'{image_id}.jpg')
        image = Image.open(path)
        if self.transform is not None:
            image = self.transform(image)

        with torch.no_grad():
            prefix = self.clip_processor.get_image_latents([image])
        if self.normalize_prefix:
            prefix = prefix.float()
            prefix = prefix / prefix.norm(2, -1)
        return tokens, mask, prefix


class MLP(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)


# сетка, связывающая CLIP и GPT
class MlpTransformer(nn.Module):
    def __init__(self, in_dim, h_dim, out_d: Optional[int] = None, act=nnf.relu, dropout=0.):
        super().__init__()
        out_d = out_d if out_d is not None else in_dim
        self.fc1 = nn.Linear(in_dim, h_dim)
        self.act = act
        self.fc2 = nn.Linear(h_dim, out_d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class MultiHeadAttention(nn.Module):

    def __init__(self, dim_self, dim_ref, num_heads, bias=True, dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_self // num_heads
        self.scale = head_dim ** -0.5
        self.to_queries = nn.Linear(dim_self, dim_self, bias=bias)
        self.to_keys_values = nn.Linear(dim_ref, dim_self * 2, bias=bias)
        self.project = nn.Linear(dim_self, dim_self)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y=None, mask=None):
        y = y if y is not None else x
        b, n, c = x.shape
        _, m, d = y.shape
        # b n h dh
        queries = self.to_queries(x).reshape(b, n, self.num_heads, c // self.num_heads)
        # b m 2 h dh
        keys_values = self.to_keys_values(y).reshape(b, m, 2, self.num_heads, c // self.num_heads)
        keys, values = keys_values[:, :, 0], keys_values[:, :, 1]
        attention = torch.einsum('bnhd,bmhd->bnmh', queries, keys) * self.scale
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            attention = attention.masked_fill(mask.unsqueeze(3), float("-inf"))
        attention = attention.softmax(dim=2)
        out = torch.einsum('bnmh,bmhd->bnhd', attention, values).reshape(b, n, c)
        out = self.project(out)
        return out, attention


class TransformerLayer(nn.Module):

    def forward_with_attention(self, x, y=None, mask=None):
        x_, attention = self.attn(self.norm1(x), y, mask)
        x = x + x_
        x = x + self.mlp(self.norm2(x))
        return x, attention

    def forward(self, x, y=None, mask=None):
        x = x + self.attn(self.norm1(x), y, mask)[0]
        x = x + self.mlp(self.norm2(x))
        return x

    def __init__(self, dim_self, dim_ref, num_heads, mlp_ratio=4., bias=False, dropout=0., act=nnf.relu,
                 norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim_self)
        self.attn = MultiHeadAttention(dim_self, dim_ref, num_heads, bias=bias, dropout=dropout)
        self.norm2 = norm_layer(dim_self)
        self.mlp = MlpTransformer(dim_self, int(dim_self * mlp_ratio), act=act, dropout=dropout)


class Transformer(nn.Module):

    def forward_with_attention(self, x, y=None, mask=None):
        attentions = []
        for layer in self.layers:
            x, att = layer.forward_with_attention(x, y, mask)
            attentions.append(att)
        return x, attentions

    def forward(self, x, y=None, mask=None):
        for i, layer in enumerate(self.layers):
            if i % 2 == 0 and self.enc_dec: # cross
                x = layer(x, y)
            elif self.enc_dec:  # self
                x = layer(x, x, mask)
            else:  # self or cross
                x = layer(x, y, mask)
        return x

    def __init__(self, dim_self: int, num_heads: int, num_layers: int, dim_ref: Optional[int] = None,
                 mlp_ratio: float = 2., act=nnf.relu, norm_layer: nn.Module = nn.LayerNorm, enc_dec: bool = False):
        super(Transformer, self).__init__()
        dim_ref = dim_ref if dim_ref is not None else dim_self
        self.enc_dec = enc_dec
        if enc_dec:
            num_layers = num_layers * 2
        layers = []
        for i in range(num_layers):
            if i % 2 == 0 and enc_dec:  # cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            elif enc_dec:  # self
                layers.append(TransformerLayer(dim_self, dim_self, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            else:  # self or cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
        self.layers = nn.ModuleList(layers)


class TransformerMapper(nn.Module):

    def forward(self, x):
        x = self.linear(x).view(x.shape[0], self.clip_length, -1)
        prefix = self.prefix_const.unsqueeze(0).expand(x.shape[0], *self.prefix_const.shape)
        prefix = torch.cat((x, prefix), dim=1)
        out = self.transformer(prefix)[:, self.clip_length:]
        return out

    def __init__(self, dim_clip: int, dim_embedding: int, prefix_length: int, clip_length: int, num_layers: int = 8):
        super(TransformerMapper, self).__init__()
        self.clip_length = clip_length
        self.transformer = Transformer(dim_embedding, 8, num_layers)
        self.linear = nn.Linear(dim_clip, clip_length * dim_embedding)
        self.prefix_const = nn.Parameter(torch.randn(prefix_length, dim_embedding), requires_grad=True)


class ClipCaptionModel(nn.Module):

    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def forward(self, tokens: torch.Tensor, prefix: torch.Tensor, mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out

    def generate_sentence(self, prefix, max_len=100):
        # greeady search
        self.eval()
        generated_sent = []
        prompt = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
        with torch.no_grad():
            while len(generated_sent) < max_len:
                _, next_token_id = self.gpt(inputs_embeds=prompt).logits[:, -1, :].max(-1)
                next_token = self.tokenizer.decode(next_token_id)
                if next_token == "<|endoftext|>":
                    break
                sys.stdout.flush()
                generated_sent.append(next_token)
                prompt = torch.cat((prompt, self.gpt.transformer.wte(next_token_id).unsqueeze(0)), dim=1)
        return generated_sent

    def __init__(self, prefix_length: int, gpt_model: str, tokenizer, clip_length: Optional[int] = None, prefix_size: int = 512,
                 num_layers: int = 8, mapping_type: MappingType = MappingType.MLP):
        super(ClipCaptionModel, self).__init__()
        self.tokenizer = tokenizer
        self.prefix_length = prefix_length
        self.gpt = GPT2LMHeadModel.from_pretrained(gpt_model)
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        if mapping_type == MappingType.MLP:
            self.clip_project = MLP((prefix_size, (self.gpt_embedding_size * prefix_length) // 2,
                                     self.gpt_embedding_size * prefix_length))
        else:
            self.clip_project = TransformerMapper(prefix_size, self.gpt_embedding_size, prefix_length,
                                                                     clip_length, num_layers)


class ClipCaptionPrefix(ClipCaptionModel):

    def parameters(self, recurse: bool = True):
        return self.clip_project.parameters()

    def train(self, mode: bool = True):
        super(ClipCaptionPrefix, self).train(mode)
        self.gpt.eval()
        return self


def save_config(args: argparse.Namespace):
    config = {}
    for key, item in args._get_kwargs():
        config[key] = item
    out_path = os.path.join(args.out_dir, f"{args.prefix}.json")
    with open(out_path, 'w') as outfile:
        json.dump(config, outfile)


def train(train_dataset: MemeDataset, val_dataset: MemeDataset, model: ClipCaptionModel, args,
          lr: float = 2e-5, warmup_steps: int = 5000, output_dir: str = ".", output_prefix: str = "", device: str ='cpu'):

    device = torch.device(device)
    batch_size = args.bs
    epochs = args.epochs
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model = model.to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=lr)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=epochs * len(train_dataloader)
    )

    for epoch in range(epochs):
        print(f">>> Training epoch {epoch}")
        train_loss = 0
        val_loss = 0
        sys.stdout.flush()
        progress = tqdm(total=len(train_dataloader), desc=output_prefix)
        # эпоха обучения
        for idx, (tokens, mask, prefix) in enumerate(train_dataloader):
            model.train()
            model.zero_grad()
            tokens, mask, prefix = tokens.to(device), mask.to(device), prefix.to(device, dtype=torch.float32)
            outputs = model(tokens, prefix, mask)
            logits = outputs.logits[:, train_dataset.prefix_length - 1: -1]
            loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            progress.set_postfix({"loss": loss.item()})
            progress.update()
            if (idx + 1) % 10000 == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(output_dir, f"{output_prefix}_latest.pt"),
                )
        progress.close()

        if epoch % args.save_every == 0 or epoch == epochs - 1:
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, f"{output_prefix}-{epoch:03d}.pt"),
            )

        # оценка на валидационном датасете
        for idx, (tokens, mask, prefix) in enumerate(val_dataloader):
            model.eval()
            with torch.no_grad():
                tokens, mask, prefix = tokens.to(device), mask.to(device), prefix.to(device, dtype=torch.float32)
                outputs = model(tokens, prefix, mask)
                logits = outputs.logits[:, train_dataset.prefix_length - 1: -1]
                loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)
                val_loss += loss.item()

        train_loss /= len(train_dataloader)
        val_loss /= len(val_dataloader)
        sys.stdout.flush()
        print(f"epoch:{epoch+1}, train_loss:{train_loss}, val_loss:{val_loss}")
        sys.stdout.flush()
    return model


def train_with_pretrain(meme_train_dataset: MemeDataset, meme_val_dataset: MemeDataset, train_dataset,
                        val_dataset, model: ClipCaptionModel, args,
          lr: float = 2e-5, warmup_steps: int = 5000, output_dir: str = ".", output_prefix: str = "", device: str ='cpu'):

    device = torch.device(device)
    batch_size = args.bs
    pretrain_epochs = int(args.extra_data_epochs)
    meme_epochs = int(args.epochs)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model = model.to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=lr)
    pretrain_train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    pretrain_val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    meme_train_dataloader = DataLoader(meme_train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    meme_val_dataloader = DataLoader(meme_val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=(meme_epochs+pretrain_epochs) * (len(pretrain_train_dataloader)+len(meme_train_dataloader))
    )
    # обучение на переведенных данных
    for epoch in range(pretrain_epochs):
        print(f">>> Training epoch {epoch}")
        train_loss = 0
        val_loss = 0
        sys.stdout.flush()
        progress = tqdm(total=len(pretrain_train_dataloader), desc=output_prefix)
        # эпоха обучения
        for idx, (tokens, mask, prefix) in enumerate(pretrain_train_dataloader):
            model.train()
            model.zero_grad()
            tokens, mask, prefix = tokens.to(device), mask.to(device), prefix.to(device, dtype=torch.float32)
            outputs = model(tokens, prefix, mask)
            logits = outputs.logits[:, args.prefix_length - 1: -1]
            loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            progress.set_postfix({"loss": loss.item()})
            progress.update()
        progress.close()

        # оценка на валидационном датасете
        for idx, (tokens, mask, prefix) in enumerate(pretrain_val_dataloader):
            model.eval()
            with torch.no_grad():
                tokens, mask, prefix = tokens.to(device), mask.to(device), prefix.to(device, dtype=torch.float32)
                outputs = model(tokens, prefix, mask)
                logits = outputs.logits[:, args.prefix_length - 1: -1]
                loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)
                val_loss += loss.item()

        train_loss /= len(pretrain_train_dataloader)
        val_loss /= len(pretrain_val_dataloader)
        print(f"epoch:{epoch+1}, train_loss:{train_loss}, val_loss:{val_loss}")
        sys.stdout.flush()

        torch.save(model.state_dict(), os.path.join(output_dir, f"{output_prefix}-pretrained-{epoch}.pt"))
        del pretrain_train_dataloader
        del pretrain_val_dataloader

    # обучение на мемах
    for epoch in range(meme_epochs):
        print(f">>> Training meme-epoch {epoch}")
        train_loss = 0
        val_loss = 0
        sys.stdout.flush()
        progress = tqdm(total=len(meme_train_dataloader), desc=output_prefix)
        # эпоха обучения
        for idx, (tokens, mask, prefix) in enumerate(meme_train_dataloader):
            model.train()
            model.zero_grad()
            tokens, mask, prefix = tokens.to(device), mask.to(device), prefix.to(device, dtype=torch.float32)
            outputs = model(tokens, prefix, mask)
            logits = outputs.logits[:, args.prefix_length - 1: -1]
            loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            progress.set_postfix({"loss": loss.item()})
            progress.update()
        progress.close()

        if epoch % args.save_every == 0 or epoch == meme_epochs - 1:
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, f"{output_prefix}-meme-{epoch+pretrain_epochs}.pt"),
            )

        # оценка на валидационном датасете
        for idx, (tokens, mask, prefix) in enumerate(meme_val_dataloader):
            model.eval()
            with torch.no_grad():
                tokens, mask, prefix = tokens.to(device), mask.to(device), prefix.to(device, dtype=torch.float32)
                outputs = model(tokens, prefix, mask)
                logits = outputs.logits[:, args.prefix_length - 1: -1]
                loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)
                val_loss += loss.item()

        train_loss /= len(meme_train_dataloader)
        val_loss /= len(meme_val_dataloader)
        sys.stdout.flush()
        print(f"epoch:{epoch+pretrain_epochs+1}, train_loss:{train_loss}, val_loss:{val_loss}")
        sys.stdout.flush()

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', default='data/for_training/train.csv')
    parser.add_argument('--val_data', default='data/for_training/val.csv')
    parser.add_argument('--images_path', default='data/images')
    parser.add_argument('--out_dir', default='models/.checkpoints')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--prefix_length', type=int, default=10)
    parser.add_argument('--prefix_length_clip', type=int, default=10)
    parser.add_argument('--bs', type=int, default=40)
    parser.add_argument('--only_prefix', dest='only_prefix', action='store_true')
    parser.add_argument('--mapping_type', type=str, default='mlp', help='mlp/transformer')
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--normalize_prefix', dest='normalize_prefix', action='store_true')
    parser.add_argument('--prefix', dest='prefix', default='mlp_meme')
    parser.add_argument('--device', dest='device', default='cuda')
    parser.add_argument('--clip-model', dest='clip_model', default='ruclip-vit-base-patch32-384')
    parser.add_argument('--gpt-model', dest='gpt_model', default='sberbank-ai/rugpt3small_based_on_gpt2')
    parser.add_argument('--extra-data', dest='extra_data', action='store_true')
    parser.add_argument('--extra-data-epochs', dest='extra_data_epochs', default=10)
    args = parser.parse_args()
    prefix_length = args.prefix_length

    DEVICE = args.device
    GPT_MODEL = args.gpt_model
    CLIP_MODEL = args.clip_model

    tokenizer = GPT2Tokenizer.from_pretrained(GPT_MODEL)

    train_dataset = MemeDataset(prefix_length, data_path=args.train_data, images_path=args.images_path,
                          normalize_prefix=args.normalize_prefix, clip_model=CLIP_MODEL, tokenizer=tokenizer)
    val_dataset = MemeDataset(prefix_length, data_path=args.val_data, images_path=args.images_path,
                          normalize_prefix=args.normalize_prefix, clip_model=CLIP_MODEL, tokenizer=tokenizer)

    train_dataset2 = torch.utils.data.ConcatDataset([
                                    MemeDataset(prefix_length, data_path='data/flickr/train_full.csv', images_path='data/flickr/images',
                          normalize_prefix=args.normalize_prefix, clip_model=CLIP_MODEL, tokenizer=tokenizer),
                                    MemeDataset(prefix_length, data_path='data/vizwiz/train.csv', images_path='data/vizwiz/train',
                          normalize_prefix=args.normalize_prefix, clip_model=CLIP_MODEL, tokenizer=tokenizer)])

    val_dataset2 = torch.utils.data.ConcatDataset([
                                    MemeDataset(prefix_length, data_path='data/flickr/val_full.csv', images_path='data/flickr/images',
                          normalize_prefix=args.normalize_prefix, clip_model=CLIP_MODEL, tokenizer=tokenizer),
                                    MemeDataset(prefix_length, data_path='data/vizwiz/val.csv', images_path='data/vizwiz/val',
                          normalize_prefix=args.normalize_prefix, clip_model=CLIP_MODEL, tokenizer=tokenizer)])

    prefix_dim = 512
    args.mapping_type = {'mlp': MappingType.MLP, 'transformer': MappingType.Transformer}[args.mapping_type]
    if args.only_prefix:
        model = ClipCaptionPrefix(prefix_length, GPT_MODEL, clip_length=args.prefix_length_clip, prefix_size=prefix_dim,
                                  num_layers=args.num_layers, mapping_type=args.mapping_type,
                                  tokenizer=tokenizer)

        print("Train only prefix")
    else:
        model = ClipCaptionModel(prefix_length, clip_length=args.prefix_length_clip, prefix_size=prefix_dim,
                                  num_layers=args.num_layers, mapping_type=args.mapping_type, gpt_model=GPT_MODEL,
                                 tokenizer=tokenizer)
        print("Train both prefix and GPT")
        sys.stdout.flush()

    if args.extra_data:
        train_with_pretrain(train_dataset, val_dataset, train_dataset2, val_dataset2, model,
                            args, output_dir=args.out_dir, output_prefix=args.prefix, device=DEVICE)
    else:
        train(train_dataset, val_dataset, model, args, output_dir=args.out_dir, output_prefix=args.prefix,
              device=DEVICE)


if __name__ == '__main__':
    main()