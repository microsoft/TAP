# Copyright (c) Facebook, Inc. and its affiliates.
from pythia.common.sample import SampleList
import random
import torch

def combine_seq(texta, textlena, textmaska, textb, textlenb, textmaskb, texta_maxlen=None):
    if textmaska is None: textmaska = torch.ones(texta.shape).long()*-1
    if textmaskb is None: textmaskb = torch.ones(textb.shape).long()*-1
    textb[0] = 102
    cmb_text, cmb_textmask = torch.cat([texta,textb],0), torch.cat([textmaska,textmaskb],0)
    cmb_textlen = textlena + textlenb
    return cmb_text, cmb_textlen, cmb_textmask

def combine_seq_pollute(batch):
    batch_size = len(batch)
    for ii in range(batch_size):
        assert(batch_size!=0)
        if batch_size!=1:
            pollute = random.choice([i for i in range(batch_size) if i!=ii])
        else:
            pollute = ii
        qidx, ocridx, objidx = ii, ii, ii
        if 'langtag_pollute' in batch[ii]: 
            if int(batch[ii].langtag_pollute)==1: qidx = pollute
        if 'ocrtag_pollute' in batch[ii]: 
            if int(batch[ii].ocrtag_pollute)==1: ocridx = pollute
        if 'objtag_pollute' in batch[ii]: 
            if int(batch[ii].objtag_pollute)==1: objidx = pollute
        qocr_text, qocr_text_len, qocr_text_mask_label = combine_seq(\
            batch[qidx].text, batch[qidx].text_len, batch[qidx].text_mask_label, \
            batch[ocridx].ocr_text, batch[ocridx].ocr_text_len, batch[ocridx].ocrtext_mask_label, texta_maxlen=batch[qidx].text.shape[0])

        batch[ii].cmb_text, batch[ii].cmb_text_len, batch[ii].cmb_text_mask_label = combine_seq(\
            qocr_text, qocr_text_len, qocr_text_mask_label, \
            batch[objidx].obj_text, batch[objidx].obj_text_len, batch[objidx].objtext_mask_label, texta_maxlen=qocr_text.shape[0])
    return batch

class BatchCollator:
    # TODO: Think more if there is a better way to do this
    _IDENTICAL_VALUE_KEYS = ["dataset_type", "dataset_name"]

    def __call__(self, batch):
        batch = combine_seq_pollute(batch)
        sample_list = SampleList(batch)
        for key in self._IDENTICAL_VALUE_KEYS:
            sample_list[key + "_"] = sample_list[key]
            sample_list[key] = sample_list[key][0]

        return sample_list