from train_baseline2 import EncoderCNN, get_loader, DecoderRNN
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import single_meteor_score
import pandas as pd
import torch
from train_baseline2 import Vocabulary
from navec import Navec
from slovnet.model.emb import NavecEmbedding

"""
Оценка бейзлайновой модели 2 на тестовых данных
"""

if __name__ == "__main__":
    vocab = Vocabulary()
    vocab.from_json("models/baseline2/baseline2_vocab.json")
    vocab_size = len(vocab)
    print(vocab_size)
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
            m += single_meteor_score(h[0], r[0][0])
        m /= len(hypotheses)

        return b1, b2, b3, b4, m

    # Initialize the encoder and decoder
    all_words = list(vocab.word2idx.keys())
    emb_path = 'navec_hudlit_v1_12B_500K_300d_100q.tar'
    navec = Navec.load(emb_path)
    ids = [navec.vocab.get(_, navec.vocab.unk_id) for _ in all_words]
    emb = NavecEmbedding(navec)
    pretrained_embeddings = emb(torch.tensor(ids))
    embed_size = 300
    hidden_size = 512
    encoder = EncoderCNN(embed_size)
    decoder = DecoderRNN(embed_size, pretrained_embeddings, hidden_size, vocab_size)
    encoder.load_state_dict(torch.load("models/baseline2/encoder_ep40.dth"))
    decoder.load_state_dict(torch.load("models/baseline2/decoder_ep40.dth"))

    encoder.to(DEVICE)
    decoder.to(DEVICE)

    decoder.eval()
    encoder.eval()

    test_dataloader = get_loader(['data/memes/images'], ['data/memes/test.csv'], vocab, shuffle=False, mode='test')

    references, hypotheses = [], []
    for x in test_dataloader:
        with torch.no_grad():
            images, captions, lengths = x
            for image, caption, caption_len in zip(images, captions, lengths):
                ref = decoder.sample(encoder(image.unsqueeze(0).to(DEVICE)), max_len=100)
                caption = covert_idx_to_sent(caption, tensor=True)
                references.append([caption])
                ref = covert_idx_to_sent(ref, tensor=False)
                hypotheses.append(ref)
    b1, b2, b3, b4, m = eval_prediction(hypotheses, references)
    print(f"test_bleu-4(meme): {b4}")

    log_file = 'output/baseline22.log'
    with open(log_file, 'a') as f:
        f.write(f"Test meme evaluation: \n")
        f.write(f"test_bleu-1(meme): {b1}\n")
        f.write(f"test_bleu-2(meme): {b2}\n")
        f.write(f"test_bleu-3(meme): {b3}\n")
        f.write(f"test_bleu-4(meme): {b4}\n")
        f.write(f"test_meteor-4(meme): {m}\n")

    test_predictions = 'output/baseline2_predictions.csv'
    with open(test_predictions, 'w') as f:
        for hyp, ref in zip(hypotheses, references):
            sent1 = " ".join(hyp)
            sent2 = " ".join(ref[0])
            f.write(f"{sent2} | {sent1}\n")

    ## плюс оценим
