import re
from transformers import AutoTokenizer
from utils.mismatched_utils import *
from src.dataset import Seq2EditVocab, MyCollate
from utils.helpers import INCORRECT_LABEL, KEEP_LABEL, PAD_LABEL, START_TOKEN, UNK_LABEL, get_target_sent_by_edits
from src.model import GECToRModel
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score
from tqdm import tqdm
import torch
import os
from utils.common_utils import init_dataloader, read_config, torch_distributed_master_process_first
from nltk.translate.bleu_score import sentence_bleu
from argparse import ArgumentParser

class Tester:
    def __init__(self):
        self.device = "cuda"
        print("self.device:", self.device)
        self.max_len = 256
        self.min_error_probability = 0.0
        self.max_pieces_per_token = 5
        self.vocab = Seq2EditVocab(
           "./data/vocabulary/d_tags.txt", "./data/vocabulary/labels_vi.txt")
        self.base_tokenizer = AutoTokenizer.from_pretrained(
            "vinai/phobert-base", use_fast=False)
        
        if bool(1):  # for roberta
            self.base_tokenizer.add_tokens([START_TOKEN], special_tokens=True)
            self.base_tokenizer_vocab = self.base_tokenizer.get_vocab()
            self.base_tokenizer_vocab[START_TOKEN] = self.base_tokenizer.unk_token_id
        self.mismatched_tokenizer = MisMatchedTokenizer(
            self.base_tokenizer,  self.base_tokenizer_vocab , self.max_len, self.max_pieces_per_token)
        self.collate_fn = MyCollate(
            input_pad_id=self.base_tokenizer.pad_token_id,
            detect_pad_id=self.vocab.detect_vocab["tag2id"][PAD_LABEL],
            correct_pad_id=self.vocab.correct_vocab["tag2id"][PAD_LABEL])
        self.model = self.init_model()
        self.model.eval()
        self.test_loader = init_dataloader(
                subset="test",
                data_path="./gec_private_train_data/testdata.edits",
                use_cache=1,
                num_workers=0,
                tokenizer=self.mismatched_tokenizer,
                vocab=self.vocab,
                input_pad_id=self.base_tokenizer.pad_token_id,
                detect_pad_id=self.vocab.detect_vocab["tag2id"][PAD_LABEL],
                correct_pad_id=self.vocab.correct_vocab["tag2id"][PAD_LABEL],
                max_len=self.max_len,
                batch_size=int(32),
                tag_strategy="keep_one",
                skip_complex=0,
                skip_correct=0,
                tp_prob=1,
                tn_prob=1)
        print("test set: ", len(self.test_loader.dataset))
    def init_model(self):
        model = GECToRModel(
            encoder_path="vinai/phobert-base",
            num_detect_tags=len(self.vocab.detect_vocab["id2tag"]),
            num_correct_tags=len(self.vocab.correct_vocab["id2tag"]),
            additional_confidence=0.0,
            dp_rate=0.0,
            detect_pad_id=self.vocab.detect_vocab["tag2id"][PAD_LABEL],
            correct_pad_id=self.vocab.correct_vocab["tag2id"][PAD_LABEL],
            detect_incorrect_id=self.vocab.detect_vocab["tag2id"][INCORRECT_LABEL],
            correct_keep_id=self.vocab.correct_vocab["tag2id"][KEEP_LABEL],
            sub_token_mode="average",
            device=self.device
        )
        checkpoint_path = os.path.join('./ckpts', 'model')
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(self.device))
        model.load_state_dict(checkpoint['state_dict'])
        model.to(self.device)

        return model
    
   
    def test_detect_tag(self):
        all_pred_d_tags = list()
        all_gold_d_tags = list()
        with torch.no_grad():
            for batch in tqdm(self.test_loader):
                for k, v in batch.items():
                    batch[k] = v.to(self.device)
                #print("batch_valid_epoch:", batch)
                outputs = self.model(batch)
                batch_word_mask = batch["word_mask"].cpu().bool()
                batch_pred_d_tags_probs = outputs["class_probabilities_d_tags"].detach(
                ).cpu()
                batch_pred_d_tags = torch.argmax(
                    batch_pred_d_tags_probs, dim=-1)
                batch_pred_d_tags = torch.masked_select(
                    batch_pred_d_tags, batch_word_mask).tolist()
                all_pred_d_tags.extend(batch_pred_d_tags)
                batch_gold_labels = torch.masked_select(
                    batch["detect_tag_ids"].cpu(), batch_word_mask).tolist()
                all_gold_d_tags.extend(batch_gold_labels)
            #print("all_gold_labels:", all_gold_labels)
            #print("all_pred_labels:",all_pred_labels)
            acc = accuracy_score(all_gold_d_tags, all_pred_d_tags)
            precision  = precision_score(all_gold_d_tags, all_pred_d_tags)
            recall  = recall_score(all_gold_d_tags, all_pred_d_tags)
            f1  = f1_score(all_gold_d_tags, all_pred_d_tags)
            print(f"acc:{acc},precision:{precision},recall:{recall},f1:{f1}")

        return acc
    def read_file(self):
        target = []
        result = []
        with open("./result/yaclc-minimal_testA.preds", 'r', encoding='utf-8') as file:
            for line in file:
                sentence = line.strip().split()
                result.append(sentence)
        with open("./datatest/target_test.txt", 'r', encoding='utf-8') as file:
            for line in file:
                sentence = line.strip().split()
                target.append(sentence)
        return target, result
    
    def bleu(self):
        target, result = self.read_file()
       # print("target:", target)
       # print("result:", result)
        bleu_scores = []
        for i in range(len(target)):
            score = sentence_bleu([target[i]], result[i], weights=(1,0,0,0))
            bleu_scores.append(score)
        average_bleu_score = sum(bleu_scores)/len(bleu_scores)
        print("BLEU score:", average_bleu_score)

def main():
    tester = Tester()
    tester.test_detect_tag()
#    tester.bleu()

if __name__ == "__main__":
    main()
