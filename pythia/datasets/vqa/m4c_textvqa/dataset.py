# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import torch
import random, math

from pythia.datasets.vqa.textvqa.dataset import TextVQADataset
from pythia.utils.text_utils import word_tokenize
from pythia.common.sample import Sample
from pythia.utils.objects_to_byte_tensor import enc_obj2bytes


class M4CTextVQADataset(TextVQADataset):
    def __init__(self, dataset_type, imdb_file_index, config, *args, **kwargs):
        super().__init__(
            dataset_type, imdb_file_index, config, *args, **kwargs
        )
        self._name = "m4c_textvqa"
        self.object_clsname = [x.strip() for x in list(open('data/1600-400-20/objects_vocab.txt','r'))]
        self.object_clsname = ['background'] + self.object_clsname
        ## set mode by checking feature path (rosetta/msocr)
        self.msocr = 'ocr_en_frcn' not in config.image_features.train[0]
        if self.msocr: assert('fc6' not in config.image_features.train[0])
        if 'pretrain' in self.config:
            self.pretrain = self.config.pretrain
        else:
            self.pretrain = False

        if 'textcap_pretrain' in self.config:
            self.textcap_pretrain=self.config.textcap_pretrain
        else:
            self.textcap_pretrain=False
        if self.pretrain and self._dataset_type=='train' and self.textcap_pretrain:
            import json
            from collections import OrderedDict
            self.textcap = OrderedDict()
            capjson = json.load(open('data/original_dl/TextCaps_0.1_train.json','r'))
            for ii in range(len(capjson['data'])):
                imid, cap, refcap = capjson['data'][ii]['image_id'], capjson['data'][ii]['caption_str'],capjson['data'][ii]['reference_strs']
                if imid in self.textcap:
                    self.textcap[imid].append(cap)
                else:
                    self.textcap[imid] = [cap]

    def preprocess_sample_info(self, sample_info):
        return sample_info  # Do nothing

    def postprocess_evalai_entry(self, entry):
        return entry  # Do nothing

    def format_for_evalai(self, report):
        answer_processor = self.answer_processor

        batch_size = len(report.question_id)
        pred_answers = report.scores.argmax(dim=-1).view(batch_size, -1)
        answer_space_size = answer_processor.get_true_vocab_size()

        predictions = []
        for idx, question_id in enumerate(report.question_id):
            # collect VQA answers
            context_tokens = report.context_tokens[idx]
            answer_words = []
            pred_source = []
            for answer_id in pred_answers[idx].tolist():
                if answer_id >= answer_space_size:
                    answer_id -= answer_space_size
                    answer_words.append(
                        word_tokenize(context_tokens[answer_id])
                    )
                    pred_source.append('OCR')
                else:
                    if answer_id == answer_processor.EOS_IDX:
                        break
                    answer_words.append(
                        answer_processor.answer_vocab.idx2word(answer_id)
                    )
                    pred_source.append('VOCAB')
            # join all the answer tokens with space
            # (this should be correct for almost all cases)
            pred_answer = ' '.join(answer_words).replace(" 's", "'s")
            entry = {
                "question_id": question_id.item(),
                "image_id": report.image_id[idx],
                "answer": pred_answer,
                "pred_source": pred_source,
            }
            entry = self.postprocess_evalai_entry(entry)

            predictions.append(entry)

        return predictions

    def load_item(self, idx):
        sample_info = self.imdb[idx]
        sample_info = self.preprocess_sample_info(sample_info)
        current_sample = Sample()

        # breaking change from VQA2Dataset: load question_id
        current_sample.question_id = torch.tensor(
            sample_info["question_id"], dtype=torch.int
        )

        if isinstance(sample_info["image_id"], int):
            current_sample.image_id = str(sample_info["image_id"])
        else:
            current_sample.image_id = sample_info["image_id"]

        if self._use_features is True:
            features = self.features_db[idx]
            if 'object_tokens' not in features['image_info_0']:
                features['image_info_0']['object_tokens'] = \
                    [self.object_clsname[x] for x in features['image_info_0']['objects']]
            current_sample.update(features)
        current_sample = self.add_sample_details(sample_info, current_sample)
        current_sample = self.add_answer_info(sample_info, current_sample)

        # only the 'max_features' key is needed
        # pop other keys to minimize data loading overhead
        for k in list(current_sample.image_info_0):
            if k == 'conf' or k == 'num_boxes':
                current_sample.image_info_0.pop(k)
        return current_sample

    def add_sample_details(self, sample_info, sample):
        ocr_str_len, obj_str_len = 100, 50
        # ocr_str_len, obj_str_len = 100, 100   ## oom on 16g

        #######################################################################
        # 1. Load text (question words)
        # breaking change from VQA2Dataset:
        # load the entire question string, not tokenized questions, since we
        # switch to BERT tokenizer in M4C and do online tokenization
        question_str = (
            sample_info['question'] if 'question' in sample_info
            else sample_info['question_str']
        )

        ## add captioning data pretrain as question
        if self.pretrain and self._dataset_type=='train' and random.random()<0.5 and self.textcap_pretrain:
            if sample_info['image_id'] in self.textcap: ## exclude stvqa samples
                question_str = self.textcap[sample_info['image_id']][random.randint(0,len(self.textcap[sample_info['image_id']])-1)]
        ## end of caption data pretrain

        processed_question = self.text_processor({"question": question_str}, updatelen=20)
        if self.pretrain:
            processed_question, output_label = self.random_word(processed_question, self.text_processor.bert_tokenizer.vocab)
            sample.text_mask_label = output_label
        else:
            sample.text_mask_label = None
        sample.text = processed_question['token_inds']
        sample.text_len = processed_question['token_num']   ## text_len include CLS, SEP
        #######################################################################
        # 2. Load object
        # object bounding box information
        obj_bbox = sample['image_info_0']['bbox']* [1./sample_info["image_width"], 1./sample_info["image_height"], 1./sample_info["image_width"], 1./sample_info["image_height"]]
        obj_max_len = self.config.processors.copy_processor.params.obj_max_length
        sample.obj_bbox_coordinates = self.copy_processor(
            {"blob": np.array(obj_bbox, dtype=np.float32)}
        )["blob"][:obj_max_len]
        ## Load OCR bert token
        obj_str = ' '.join(sample['image_info_0']['object_tokens'])
        # obj_str = ' '.join([x for x in sample['image_info_0']['object_tokens'] if x != 'background'])
        processed_obj = self.text_processor({"question": obj_str}, updatelen=obj_str_len)
        if self.pretrain:
            processed_obj, output_label = self.random_word(processed_obj, self.text_processor.bert_tokenizer.vocab)#, mask_prob=0.)
            sample.objtext_mask_label = output_label
        else:
            sample.objtext_mask_label = None
        sample.obj_text = processed_obj['token_inds']
        sample.obj_text_len = processed_obj['token_num']
        #######################################################################
        # 3. Load OCR
        if not self.use_ocr:
            # remove all OCRs from the sample
            # (i.e. make an empty OCR list)
            sample_info['ocr_tokens'] = []
            sample_info['ocr_info'] = []
            if 'ocr_normalized_boxes' in sample_info:
                sample_info['ocr_normalized_boxes'] = np.zeros(
                    (0, 4), np.float32
                )
            # clear OCR visual features
            sample.image_feature_1 = torch.zeros_like(sample.image_feature_1)
        # Preprocess OCR tokens
        ## ocr_token_processor for lower(), etc.
        if self.msocr:   ## load ms ocr prediction directly from _info.npy; else: from imdb provided
            max_len = self.config.processors.answer_processor.params.max_length
            if 'ocr_conf' not in sample.image_info_1: sample.image_info_1['ocr_conf']=None
            if len(sample.image_info_1["ocr_tokens"])>max_len:
                sample.image_info_1['ocr_tokens'], sample.image_info_1['ocr_boxes'] = \
                    self.ocr_truncate(np.array(sample.image_info_1['ocr_tokens']), sample.image_info_1['ocr_boxes'], \
                        conf_array=np.array(sample.image_info_1['ocr_conf']), mode='naive')
            ocr_tokens = [
                self.ocr_token_processor({"text": token})["text"]
                for token in sample.image_info_1["ocr_tokens"]]
        else:
            print('not using msocr; duplicate warning; please manually comment this line')
            ocr_tokens = [
                self.ocr_token_processor({"text": token})["text"]
                for token in sample_info["ocr_tokens"]]
        ## Load OCR bert token
        ## might lead to "Token indices sequence length is longer than the specified maximum sequence length for this model" if too long
        ## but truncated to updatelen anyway
        ocr_str = ' '.join(ocr_tokens)
        processed_ocr = self.text_processor({"question": ocr_str}, updatelen=ocr_str_len)

        # processed_ocr = self.text_processor({"question": ocr_str}, updatelen=20)
        if self.pretrain:
            processed_ocr, output_label = self.random_word(processed_ocr, self.text_processor.bert_tokenizer.vocab)
            sample.ocrtext_mask_label = output_label
            ## 0 is not pollute
            rand = random.random()
            sample.ocrtag_pollute = torch.tensor([rand<0.5]).long()
            sample.langtag_pollute = torch.tensor([rand>1.]).long()
            sample.objtag_pollute = torch.tensor([rand>1.]).long()
            sample.tag_pollute = (sample.ocrtag_pollute + sample.langtag_pollute + sample.objtag_pollute).clamp(max=1)
        else:
            sample.ocrtext_mask_label = None
        sample.ocr_text = processed_ocr['token_inds']
        sample.ocr_text_len = processed_ocr['token_num']
        #######################################################################
        # Get FastText embeddings for OCR tokens
        context = self.context_processor({"tokens": ocr_tokens})
        sample.context = context["text"]
        sample.context_tokens = context["tokens"]
        sample.context_tokens_enc = enc_obj2bytes(context["tokens"])
        # sample.context_tokens_enc = enc_obj2bytes(context["tokens"], max_size=4094*2)
        sample.context_feature_0 = context["text"]
        sample.context_info_0 = Sample()
        sample.context_info_0.max_features = context["length"]
        # Get PHOC embeddings for OCR tokens
        context_phoc = self.phoc_processor({"tokens": ocr_tokens})
        sample.context_feature_1 = context_phoc["text"]
        sample.context_info_1 = Sample()
        sample.context_info_1.max_features = context_phoc["length"]
        # OCR order vectors
        # TODO remove order_vectors -- it is no longer needed in M4C
        order_vectors = np.eye(len(sample.context_tokens), dtype=np.float32)
        order_vectors = torch.from_numpy(order_vectors)
        order_vectors[context["length"]:] = 0
        sample.order_vectors = order_vectors
        # OCR bounding box information
        if self.msocr:
            max_len = self.config.processors.answer_processor.params.max_length
            ocr_bbox = np.zeros((0,4),dtype=np.float32)
            if sample.image_info_1['ocr_boxes'].shape[0]!=0:
                ocr_bbox = sample.image_info_1['ocr_boxes']* [1./sample_info["image_width"], 1./sample_info["image_height"], 1./sample_info["image_width"], 1./sample_info["image_height"]]
            sample.ocr_bbox_coordinates = self.copy_processor(
                {"blob": np.array(ocr_bbox, dtype=np.float32)}
            )["blob"][:max_len]
        else:
            if 'ocr_normalized_boxes' in sample_info:
                # New imdb format: OCR bounding boxes are already pre-computed
                max_len = self.config.processors.answer_processor.params.max_length
                sample.ocr_bbox_coordinates = self.copy_processor(
                    {"blob": sample_info['ocr_normalized_boxes']}
                )["blob"][:max_len]
            else:
                # Old imdb format: OCR bounding boxes are computed on-the-fly
                # from ocr_info
                sample.ocr_bbox_coordinates = self.bbox_processor(
                    {"info": sample_info["ocr_info"]}
                )["bbox"].coordinates

        #######################################################################
        ## RPP objective
        ## add OCR, OBJ region spatial location info
        ## If OCR region falls inside OBJ
        sample.overlap = self.region_overlap(sample.obj_bbox_coordinates, sample.ocr_bbox_coordinates[:,:4], threshold=0.99)
        targetoverlap = torch.tensor(0) if random.random()<0.5 else torch.tensor(1)
        index = torch.where(sample.overlap==targetoverlap)
        len_index = int(index[0].shape[0])
        if len_index!=0:
            ind = random.randint(0,len_index-1)
            sample.overlap = targetoverlap
            sample.overlap_obj, sample.overlap_ocr = index[0][ind],index[1][ind]
        else:
            sample.overlap = torch.tensor(-1)
            sample.overlap_obj, sample.overlap_ocr = torch.tensor(0), torch.tensor(0)
        return sample

    def add_answer_info(self, sample_info, sample):
        sample_has_answer = ("answers" in sample_info)
        if sample_has_answer:
            # Load real answers from sample_info
            answers = sample_info["answers"]
            sample.gt_answers_enc = enc_obj2bytes(answers)
            answer_processor_arg = {
                "answers": answers,
                "context_tokens": sample.context_tokens,
            }
            processed_answers = self.answer_processor(answer_processor_arg)

            assert not self.config.fast_read, \
                'In M4CTextVQADataset, online OCR sampling is incompatible ' \
                'with fast_read, so fast_read is currently not supported.'
            sample.targets = processed_answers["answers_scores"]
            sample.sampled_idx_seq = processed_answers["sampled_idx_seq"]
            sample.train_prev_inds = processed_answers["train_prev_inds"]
            sample.train_loss_mask = processed_answers["train_loss_mask"]
        else:
            # Load dummy answers as placeholders
            answer_params = self.config.processors.answer_processor.params
            sample.sampled_idx_seq = None
            sample.train_prev_inds = torch.zeros(
                answer_params.max_copy_steps, dtype=torch.long
            )

        return sample

    def ocr_truncate(self, token_array, ocr_boxes, conf_array=None, mode='naive'):
        max_len = self.config.processors.answer_processor.params.max_length
        if conf_array is None: mode='naive'
        ## default original order
        if 'naive' in mode:
            idx = np.array(range(ocr_boxes.shape[0]))[:max_len]
        return token_array[idx].tolist(), ocr_boxes[idx,:]

    def region_overlap(self, bbox_obj, bbox_ocr, threshold=0.99, bgimgsize_thes=0.25, bgobj_mask=None, eps=1e-6):
        ## (x1, y1, x2, y2)
        bboxmask_obj = (bbox_obj.sum(1)!=0).long().sum()
        bboxmask_ocr = (bbox_ocr.sum(1)!=0).long().sum()
        area_obj = (bbox_obj[:, 2] - bbox_obj[:, 0]) * (bbox_obj[:, 3] - bbox_obj[:, 1])
        area_ocr = (bbox_ocr[:, 2] - bbox_ocr[:, 0]) * (bbox_ocr[:, 3] - bbox_ocr[:, 1])

        lt = torch.max(bbox_obj[:, None, :2], bbox_ocr[:, :2])  # [N,M,2]
        rb = torch.min(bbox_obj[:, None, 2:], bbox_ocr[:, 2:])  # [N,M,2]

        wh = (rb - lt).clamp(min=0)  # [N,M,2]
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

        cover = area_obj.unsqueeze(1).repeat(1,area_ocr.shape[0]) / (area_obj[:, None] + area_ocr - inter)
        cover = (cover>threshold).long()
        cover[bboxmask_obj:,:]=-1
        cover[:,bboxmask_ocr:]=-1
        if bgobj_mask is None:
            bgobj_mask = (area_obj>bgimgsize_thes)
        cover[bgobj_mask,:]=-1
        return cover

    """
    From VilBert, dataset/concept_cap_dataset
    """
    def random_word(self, tokens, tokenizer, mask_prob=0.15):
        """
        Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
        :param tokens: list of str, tokenized sentence.
        :param tokenizer: Tokenizer, object used for tokenization (we need it's vocab here)
        :return: (list of str, list of int), masked tokens and related labels for LM prediction
        """
        assert tokenizer['[MASK]'] == 103
        assert tokenizer['[UNK]'] == 100
        mask_prob = mask_prob

        output_label = []
        token_inds = tokens['token_inds']
        for i, token in enumerate(token_inds):
            prob = random.random()
            # mask token with 15% probability
            
            if prob < mask_prob and token not in [0,101,102]:
                # append current token to output (we will predict these later)
                try:
                    # output_label.append(tokenizer.vocab[token])
                    output_label.append(int(token))
                except KeyError:
                    # For unknown words (should not occur with BPE vocab)
                    # output_label.append(tokenizer.vocab["[UNK]"])
                    output_label.append(100)
                    logger.warning(
                        "Cannot find token '{}' in vocab. Using [UNK] insetad".format(token)
                    )

                prob /= mask_prob

                # 80% randomly change token to mask token
                if prob < 0.8:
                    # token_inds[i] = "[MASK]"
                    token_inds[i] = 103

                # 10% randomly change token to random token
                elif prob < 0.9:
                    # token_inds[i] = random.choice(list(tokenizer.vocab.items()))[0]
                    token_inds[i] = random.choice(list(range(1000,len(tokenizer))))#[0]

                # -> rest 10% randomly keep current token

            else:
                # no masking token (will be ignored by loss function later)
                output_label.append(-1)
        tokens['token_inds'] = token_inds
        return tokens, torch.tensor(output_label)

    def random_word_answer(self, processed_answers, mask_prob=0.15):
        pretrain_targets = processed_answers["answers_scores"] * 0
        pretrain_prev_inds = processed_answers["train_prev_inds"]
        pretrain_loss_mask = processed_answers["train_loss_mask"] * 0

        for ii in range(pretrain_prev_inds.shape[0]):
            prob = random.random()
            if prob<mask_prob and pretrain_prev_inds[ii]!=0 and pretrain_prev_inds[ii]!=1:
                pretrain_loss_mask[ii] = 1
                pretrain_targets[ii,:] = processed_answers["answers_scores"][ii-1,:]
                prob /= mask_prob
                if prob < 0.8:
                    pretrain_prev_inds[ii] = 3  ## <unk>
                elif prob < 0.9:
                    pretrain_prev_inds[ii] = random.choice(list(range(4,pretrain_targets.shape[1]-1)))
                else:
                    pretrain_loss_mask[ii] = 0
        processed_answers["answers_scores"] = pretrain_targets
        processed_answers["train_prev_inds"] = pretrain_prev_inds
        processed_answers["train_loss_mask"] = pretrain_loss_mask
        return processed_answers