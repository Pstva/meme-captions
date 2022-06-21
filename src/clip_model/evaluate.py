from train import ClipCaptionModel, ClipCaptionPrefix, MemeDataset, MappingType
import argparse
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import single_meteor_score
from nltk import word_tokenize
import sys
import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer


def eval_prediction(hypotheses, references):

    references = [[word_tokenize(x)] for x in references]
    hypotheses = [word_tokenize(x) for x in hypotheses]

    # bleu-score
    b1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))
    b2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5))
    b3 = corpus_bleu(references, hypotheses, weights=(1 / 3, 1 / 3, 1 / 3))
    b4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))

    m = 0
    for h, r in zip(hypotheses, references):
        m += single_meteor_score(r[0], h)
    m /= len(hypotheses)

    return b1, b2, b3, b4, m


def eval(val_dataset, model, args, device):

    device = torch.device(device)
    batch_size = args.bs
    model.to(device)
    model.train()
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    generated_captions = []

    # оценка на валидационном датасете
    for idx, (tokens, mask, prefix) in enumerate(val_dataloader):
        model.eval()
        with torch.no_grad():
            tokens, mask, prefix = tokens.to(device), mask.to(device), prefix.to(device, dtype=torch.float32)
            for p in prefix:
                s = "".join(model.generate_sentence(p, max_len=50))
                generated_captions.append(s)

    print("real | generated")
    for r, c in zip(val_dataset.captions, generated_captions):
        print(f"{r} | {c}")

    print()

    sys.stdout.flush()

    b1, b2, b3, b4, m = eval_prediction(generated_captions, val_dataset.captions)
    print(f"Test meme evaluation:")
    print(f"test_bleu-1(meme): {b1}")
    print(f"test_bleu-2(meme): {b2}")
    print(f"test_bleu-3(meme): {b3}")
    print(f"test_bleu-4(meme): {b4}")
    print(f"test_meteor-4(meme): {m}")
    print()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data', default='data/for_training/val.csv')
    parser.add_argument('--images_path', default='data/images')
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
    parser.add_argument('--model-path', default='models/.checkpoints/mlp_meme-002.pt')
    args = parser.parse_args()
    prefix_length = args.prefix_length

    DEVICE = args.device
    GPT_MODEL = args.gpt_model
    CLIP_MODEL = args.clip_model

    tokenizer = GPT2Tokenizer.from_pretrained(GPT_MODEL)

    val_dataset = MemeDataset(prefix_length, data_path=args.test_data, images_path=args.images_path,
                          normalize_prefix=args.normalize_prefix, clip_model=CLIP_MODEL, tokenizer=tokenizer)
    prefix_dim = 512
    args.mapping_type = {'mlp': MappingType.MLP, 'transformer': MappingType.Transformer}[args.mapping_type]

    if args.only_prefix:
        model = ClipCaptionPrefix(prefix_length, GPT_MODEL, clip_length=args.prefix_length_clip, prefix_size=prefix_dim,
                                  num_layers=args.num_layers, mapping_type=args.mapping_type,
                                  tokenizer=tokenizer)
    else:
        model = ClipCaptionModel(prefix_length, clip_length=args.prefix_length_clip, prefix_size=prefix_dim,
                                  num_layers=args.num_layers, mapping_type=args.mapping_type, gpt_model=GPT_MODEL,
                                 tokenizer=tokenizer)

    model.load_state_dict(torch.load(args.model_path))
    eval(val_dataset, model, args, device=DEVICE)

if __name__ == "__main__":
    main()

