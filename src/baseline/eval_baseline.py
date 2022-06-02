from train_baseline import EncoderCNN, DecoderRNN, build_vocab, get_loader
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import single_meteor_score
import pandas as pd
import torch
"""
Оценка бейзлайновой модели на тестовых данных
"""

if __name__ == "__main__":
    captions = list(pd.read_csv('data/flickr/train.csv', sep='\t')['text'])
    captions3 = list(pd.read_csv('data/for_training/train.csv', sep='\t')['text'])
    captions.extend(captions3)
    vocab = build_vocab(captions)
    vocab_size = len(vocab)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def covert_idx_to_sent(idx_caption, tensor=True):
        sent = []
        for x in idx_caption:
            if x == 0 or x == 2:
                return sent
            if tensor:
                sent.append(vocab.idx2word[x.item()])
            else:
                sent.append(vocab.idx2word[x])
        return sent


    def eval_prediction(hypotheses, references):
        # bleu-score
        b1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))
        b2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5))
        b3 = corpus_bleu(references, hypotheses, weights=(1 / 3, 1 / 3, 1 / 3))
        b4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))

        # meteor score
        m = 0
        for h, r in zip(hypotheses, references):
            m += single_meteor_score(h, r[0])
        m /= len(hypotheses)

        return b1, b2, b3, b4, m

    # Initialize the encoder and decoder
    embed_size = 512
    hidden_size = 512
    encoder = EncoderCNN(embed_size)
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
    encoder.load_state_dict(torch.load("models/baseline_encoder.pth"))
    decoder.load_state_dict(torch.load("models/baseline_decoder.pth"))

    device = DEVICE
    encoder.to(device)
    decoder.to(device)

    decoder.eval()
    encoder.eval()

    test_dataloader = get_loader('data/images', 'data/for_training/test.csv', vocab, shuffle=False, mode='test')

    references, hypotheses = [], []
    for x in test_dataloader:
        with torch.no_grad():
            images, captions, lengths = x
            for image, caption, caption_len in zip(images, captions, lengths):
                ref = decoder.sample(encoder(image.unsqueeze(0).to(DEVICE)))
                caption = covert_idx_to_sent(caption, tensor=True)
                references.append([caption])
                ref = covert_idx_to_sent(ref, tensor=False)
                hypotheses.append(ref)
    b1, b2, b3, b4, m = eval_prediction(hypotheses, references)
    print(f"test_bleu-4(meme): {b4}")

    log_file = 'output/baseline.log'
    with open(log_file, 'a') as f:
        f.write(f"Test meme evaluation: \n")
        f.write(f"test_bleu-1(meme): {b1}\n")
        f.write(f"test_bleu-2(meme): {b2}\n")
        f.write(f"test_bleu-3(meme): {b3}\n")
        f.write(f"test_bleu-4(meme): {b4}\n")
        f.write(f"test_meteor-4(meme): {m}\n")

    test_predictions = 'output/baseline_predictions.csv'
    with open(test_predictions, 'w') as f:
        for hyp, ref in zip(hypotheses, references):
            sent1 = " ".join(hyp)
            sent2 = " ".join(ref[0])
            f.write(f"{sent2}\t{sent1}\n")


#### predictions for flickr

    test_dataloader = get_loader('data/flickr/images', 'data/flickr/val.csv', vocab, shuffle=False, mode='test')

    references, hypotheses = [], []
    for x in test_dataloader:
        with torch.no_grad():
            images, captions, lengths = x
            for image, caption, caption_len in zip(images, captions, lengths):
                ref = decoder.sample(encoder(image.unsqueeze(0).to(DEVICE)))
                caption = covert_idx_to_sent(caption, tensor=True)
                references.append([caption])
                ref = covert_idx_to_sent(ref, tensor=False)
                hypotheses.append(ref)
    b1, b2, b3, b4, m = eval_prediction(hypotheses, references)
    print(f"test_bleu-4(meme): {b4}")
    #
    # test_predictions = 'output/baseline_predictions_flickr.csv'
    # with open(test_predictions, 'w') as f:
    #     for hyp, ref in zip(hypotheses, references):
    #         sent1 = " ".join(hyp)
    #         sent2 = " ".join(ref[0])
    #         f.write(f"{sent2}\t{sent1}\n")
