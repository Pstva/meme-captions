import json
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from nltk import word_tokenize
from collections import Counter
import os
from nltk.translate.bleu_score import corpus_bleu
import pandas as pd
from tqdm import tqdm
import numpy as np
from navec import Navec
from slovnet.model.emb import NavecEmbedding


MAX_SEQ_LEN = 100
PRETRAIN_EPOCHS = 40
MEME_EPOCHS = 20
SAVE_EVERY = 2
BATCH_SIZE = 25
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"training on {DEVICE}")
torch.manual_seed(10)

log_file = 'output/baseline22.log'
output_folder = "models/baseline2"
vocab_path_to_save = "models/baseline2/baseline2_vocab.json"

embedding_path = 'navec_hudlit_v1_12B_500K_300d_100q.tar'

embed_size = 300
hidden_size = 512

# класс для данных пары картинка (путь до картинки) - текст
class Data(Dataset):
    def __init__(self, image_path, data_path, vocab, max_len=MAX_SEQ_LEN, mode='train'):
        self.max_len = max_len
        self.image_path = image_path
        self.image_ids = []
        self.captions = []
        # загружаем id картинок и их captions
        with open(data_path, 'r') as f:
            f.readline()
            for line in f:
                uid, caption = line.strip().split('\t')
                self.captions.append(caption)
                self.image_ids.append(uid)
        self.vocab = vocab
        if mode == 'train':
            self.transform = transforms.Compose([transforms.Resize(256),
                                              transforms.RandomCrop(224),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        else:
            self.transform = transforms.Compose([transforms.Resize((224, 224)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    def __getitem__(self, index):

        caption = self.captions[index]
        img_id = self.image_ids[index]
        path = os.path.join(self.image_path, f'{img_id}.jpg')
        image = Image.open(path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        tokens = word_tokenize(str(caption).lower())
        caption = []
        # добавляем в описание токен начала
        #caption.append(self.vocab('<start>'))
        # добавялем все остальные токены
        caption.extend([self.vocab(token) for token in tokens])
        caption_len = len(caption)
        # если описание слишком длинное - обрезаем его
        if len(caption) > self.max_len:
            caption = caption[:self.max_len]
        # добавляем токен конца
        #caption.append(self.vocab('<end>'))
        # если описание недостаточно длинное - добавляем паддинги
        if len(caption) < self.max_len:
            caption.extend([self.vocab('<pad>')] * (self.max_len - len(caption)))

        target = torch.tensor(caption, dtype=torch.long)
        return image, target, caption_len

    def __len__(self):
        return len(self.image_ids)


# класс для хранения словаря слов
class Vocabulary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

    def save(self, path_to_save):
        with open(path_to_save, "w") as f:
            json.dump(self.idx2word, f)

    def from_json(self, path):
        with open(path, 'r') as f:
            self.word2idx = json.load(f)
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.idx = len(self.word2idx)


# функция для построения словаря по имеющимся текстам
def build_vocab(captions, threshold=1):
    counter = Counter()
    for caption in captions:
        tokens = word_tokenize(str(caption).lower())
        counter.update(tokens)

    words = [word for word, cnt in counter.items() if cnt >= threshold]

    vocab = Vocabulary()
    vocab.add_word('<pad>')  # 0
    vocab.add_word('<unk>')  # 1

    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab


def get_loader(images_path, datas_path, vocab, batch_size=BATCH_SIZE, shuffle=True, mode='train'):

    if len(images_path) == 1:
        dataset = Data(image_path=images_path[0], data_path=datas_path[0], vocab=vocab, mode=mode)
    datasets = []
    if len(images_path) > 1:
        for image_path, data_path in zip(images_path, datas_path):
            datasets.append(Data(image_path=image_path, data_path=data_path, vocab=vocab, mode=mode))
        dataset = torch.utils.data.ConcatDataset(datasets)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)


# модель - кодировщик картинки
class EncoderCNN(nn.Module):
    """
    достаем эмбеддинги для картинок с помощью предобученной ResNet
    """

    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.resnet.eval()
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.batch = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.batch(self.embed(features))
        return features


# Модель-декодировщик картинки
# лстм, которая принимает на вход эмьеддинг слова и вектор картинки и предсказывает следующее слово
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, pretrained_emb, hidden_size, vocab_size, num_layers=2):
        """ Initialize the layers of this model."""
        super().__init__()

        self.embed = nn.Embedding.from_pretrained(pretrained_emb)
        self.lstm = nn.LSTM(input_size=embed_size,
                            hidden_size=hidden_size,  # LSTM hidden units
                            num_layers=num_layers,  # number of LSTM layer
                            bias=True,  # use bias weights b_ih and b_hh
                            batch_first=True,  # input & output will have batch size as 1st dimension
                            dropout=0.5,
                            bidirectional=False,  # unidirectional LSTM
                            )
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def init_hidden(self, batch_size):
        """ At the start of training, we need to initialize a hidden state;
        there will be none because the hidden state is formed based on previously seen data.
        So, this function defines a hidden state with all zeroes
        The axes semantics are (num_layers, batch_size, hidden_dim)
        """
        return (torch.zeros((self.num_layers, batch_size, self.hidden_size), device=DEVICE),
                torch.zeros((self.num_layers, batch_size, self.hidden_size), device=DEVICE))

    def forward(self, features, captions):
        """ Define the feedforward behavior of the model """

        # Initialize the hidden state
        batch_size = features.shape[0]  # features is of shape (batch_size, embed_size)
        hidden = self.init_hidden(batch_size)

        # Create embedded word vectors for each word in the captions
        embeddings = self.embed(captions)  # new shape : (batch_size, captions length, embed_size)

        # Stack the features and captions
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)  # new shape : (batch_size, caption length, embed_size)

        # Get the output and hidden state by passing the lstm over our word embeddings
        # the lstm takes in our embeddings and hidden state
        lstm_out, hidden = self.lstm(embeddings, hidden)  # lstm_out shape : (batch_size, caption length, hidden_size)

        # Fully connected layer
        return self.linear(lstm_out[:, :-1, :])  # outputs shape : (batch_size, caption length, vocab_size)


    def sample(self, inputs, max_len):
        """
        генерация описания под картинку
        каждый раз слово выбирается на основании вероятностей из софтмакса
        """

        output = []
        batch_size = inputs.shape[0]
        hidden = self.init_hidden(batch_size)  # Get initial hidden state of the LSTM
        self.eval()
        while len(output) < max_len:
            with torch.no_grad():
                lstm_out, hidden = self.lstm(inputs.unsqueeze(1), hidden)  # lstm_out shape : (1, 1, hidden_size)
                outputs = self.linear(lstm_out)  # outputs shape : (1, 1, vocab_size)
                outputs = outputs.squeeze(1)  # outputs shape : (1, vocab_size)
                p = F.softmax(outputs, dim=1).detach().cpu().numpy()[0]
            max_index = np.random.choice(len(outputs[0]), p=p)  # predict the next word
            output.append(max_index)  # storing the word predicted
            # <pad> token
            if (max_index == 0):
                break

            ## Prepare to embed the last predicted word to be the new input of the lstm
            inputs = self.embed(torch.tensor(max_index).to(DEVICE))  # inputs shape : (1, embed_size)
            inputs = inputs.unsqueeze(0)
        return output


# Function for training
def train():
    # строим словарь
    captions = list(pd.read_csv('data/flickr/train_full.csv', sep='\t')['text'])
    captions2 = list(pd.read_csv('data/vizwiz/train.csv', sep='\t')['text'])
    captions3 = list(pd.read_csv('data/memes/train.csv', sep='\t')['text'])
    captions.extend(captions2)
    captions.extend(captions3)
    vocab = build_vocab(captions, threshold=5)
    vocab_size = len(vocab)
    print(f"Size of vocabulary:{vocab_size}")
    del captions
    del captions2
    del captions3


    # сохраняем словарь
    vocab.save(vocab_path_to_save)

    # подготавливаем предобученные эмбеддинги
    all_words = list(vocab.word2idx.keys())
    emb_path = embedding_path
    navec = Navec.load(emb_path)
    ids = [navec.vocab.get(_, navec.vocab.unk_id) for _ in all_words]
    emb = NavecEmbedding(navec)
    pretrained_embeddings = emb(torch.tensor(ids))

    # инициализируем кодировщик и декодировщик
    encoder = EncoderCNN(embed_size)
    decoder = DecoderRNN(embed_size, pretrained_embeddings, hidden_size, vocab_size)
    encoder.to(DEVICE)
    decoder.to(DEVICE)

    # инициализируем оптимайзер и лосс-функцию
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    params = list(decoder.parameters()) + list(encoder.embed.parameters()) + list(encoder.batch.parameters())
    optimizer = torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08)

    # сначала в течение PRETRAIN_EPOCHS обучаем сеть на датасете flickr

    train_dataloader = get_loader(['data/flickr/images', 'data/vizwiz/train'], ['data/flickr/train_full.csv',
                                                                                 'data/vizwiz/train.csv'], vocab)
    val_dataloader = get_loader(['data/flickr/images', 'data/vizwiz/val'], ['data/flickr/val_full.csv',
                                                                               'data/vizwiz/val.csv'], vocab,
                                mode='test')

    # функуия, переводящая список или тензор с индексами в предложение
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

    print("Starting training")

    for epoch in range(PRETRAIN_EPOCHS):

        train_total_loss = 0
        val_total_loss = 0

        # шаг обучения
        for x in tqdm(train_dataloader):
            decoder.train()
            encoder.train()
            images, captions, _ = x
            outputs = decoder(encoder(images.to(DEVICE)), captions.to(DEVICE))
            loss = criterion(outputs.contiguous().view(-1, vocab_size), captions.view(-1).to(DEVICE))
            train_total_loss += loss.item()
            loss.backward()
            nn.utils.clip_grad_norm_(decoder.parameters(), 5)
            nn.utils.clip_grad_norm_(encoder.embed.parameters(), 5)
            nn.utils.clip_grad_norm_(encoder.batch.parameters(), 5)
            optimizer.step()
            optimizer.zero_grad()

        train_total_loss /= len(train_dataloader)

        # шаг оценки модели, лосс на вал датасете  + bleu-3 на нем же
        hypotheses, references = [], []
        for x in tqdm(val_dataloader):
            decoder.eval()
            encoder.eval()
            with torch.no_grad():
                images, captions, lengths = x
                for image, caption, caption_len in zip(images, captions, lengths):
                    ref = decoder.sample(encoder(image.unsqueeze(0).to(DEVICE)), max_len=100)
                    caption = covert_idx_to_sent(caption, tensor=True)
                    references.append([caption])
                    ref = covert_idx_to_sent(ref, tensor=False)
                    hypotheses.append(ref)
                outputs = decoder(encoder(images.to(DEVICE)), captions.to(DEVICE))
                loss = criterion(outputs.contiguous().view(-1, vocab_size), captions.view(-1).to(DEVICE))

                val_total_loss += loss.item()

        # просто печать рандомного реального описания и нашей генерации, чтобы посмотреть какобстоят дела у сети
        print(f"val ref: {references[0]}")
        print(f"val hyp: {hypotheses[0]}")

        with open(log_file, 'a') as f:
            f.write(f"val ref: {references[0]}\n")
            f.write(f"val hyp: {hypotheses[0]}\n")

        val_total_loss /= len(val_dataloader)
        val_bleu = corpus_bleu(references, hypotheses, weights=(1/2, 1/2))
        print(f"Epoch: {epoch+1}, train_loss: {train_total_loss}, val_loss: {val_total_loss}, val_bleu-2: {val_bleu}")
        with open(log_file, 'a') as f:
            f.write(f"Epoch: {epoch+1}, train_loss: {train_total_loss}, val_loss: {val_total_loss}, val_bleu-2: {val_bleu}\n")

    torch.save(encoder.state_dict(), f"{output_folder}/encoder_ep{epoch + 1}_flickr.dth")
    torch.save(decoder.state_dict(), f"{output_folder}/decoder_ep{epoch + 1}_flickr.dth")

    # Далее, дообучаем на описаниях мемов в течение максимум MEME_EPOCHS
    # patience = PATIENCE, если в течение этого числа эпох bleu-3 не увеличивается, заканчиваем обучение

    with open(log_file, 'a') as f:
        f.write(f"Starting training meme captions\n\n")

    train_dataloader = get_loader(['data/memes/images'], ['data/memes/train.csv'], vocab)
    val_dataloader = get_loader(['data/memes/images'], ['data/memes/val.csv'], vocab, mode='test')

    for epoch in range(MEME_EPOCHS):
        epoch += PRETRAIN_EPOCHS

        train_total_loss = 0
        val_total_loss = 0

        #  шаг обучения
        for x in tqdm(train_dataloader):
            decoder.train()
            encoder.train()
            images, captions, _ = x
            outputs = decoder(encoder(images.to(DEVICE)), captions.to(DEVICE))
            loss = criterion(outputs.contiguous().view(-1, vocab_size), captions.view(-1).to(DEVICE))
            train_total_loss += loss.item()
            loss.backward()
            nn.utils.clip_grad_norm_(decoder.parameters(), 5)
            nn.utils.clip_grad_norm_(encoder.embed.parameters(), 5)
            nn.utils.clip_grad_norm_(encoder.batch.parameters(), 5)
            optimizer.step()
            optimizer.zero_grad()

        train_total_loss /= len(train_dataloader)
        # шаг оценки на валидационном датасете
        hypotheses, references = [], []
        for x in tqdm(val_dataloader):
            decoder.eval()
            encoder.eval()
            with torch.no_grad():
                images, captions, lengths = x
                for image, caption, caption_len in zip(images, captions, lengths):
                    ref = decoder.sample(encoder(image.unsqueeze(0).to(DEVICE)), max_len=100)
                    caption = covert_idx_to_sent(caption, tensor=True)
                    references.append([caption])
                    ref = covert_idx_to_sent(ref, tensor=False)
                    hypotheses.append(ref)
                outputs = decoder(encoder(images.to(DEVICE)), captions.to(DEVICE))
                loss = criterion(outputs.contiguous().view(-1, vocab_size), captions.view(-1).to(DEVICE))

                val_total_loss += loss.item()

        print(f"val ref: {references[0]}")
        print(f"val hyp: {hypotheses[0]}")

        with open(log_file, 'a') as f:
            f.write(f"meme val ref: {references[0]}\n")
            f.write(f"meme val hyp: {hypotheses[0]}\n")


        val_total_loss /= len(val_dataloader)
        val_bleu = corpus_bleu(references, hypotheses, weights=(1/3, 1/3, 1/3))
        print(f"Epoch: {epoch+1}, train_loss: {train_total_loss}, val_loss: {val_total_loss}, val_bleu-3(meme): {val_bleu}")

        with open(log_file, 'a') as f:
            f.write(f"Epoch: {epoch+1}, train_loss: {train_total_loss}, val_loss: {val_total_loss}, val_bleu-3(meme): {val_bleu}\n")

        if (epoch + 1) % SAVE_EVERY == 0:
            torch.save(encoder.state_dict(), f"{output_folder}/encoder_ep{epoch+1}.dth")
            torch.save(decoder.state_dict(), f"{output_folder}/decoder_ep{epoch+1}.dth")


if __name__ == "__main__":
    train()
