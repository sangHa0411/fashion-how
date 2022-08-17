import re
import pickle
import codecs
import numpy as np

class SubWordEmbReaderUtil:
    """
    Class for subword embedding
    """
    def __init__(self, data_path):
        """
        initialize
        """
        with open(data_path, 'rb') as fp:
            self._subw_length_min = pickle.load(fp)
            self._subw_length_max = pickle.load(fp)
            self._subw_dic = pickle.load(fp, encoding='euc-kr')
            self._emb_np = pickle.load(fp, encoding='bytes')
            self._emb_size = self._emb_np.shape[1]

    def get_emb_size(self):
        """
        get embedding size
        """
        return self._emb_size

    def _normalize_func(self, s):
        """
        normalize
        """
        s1 = re.sub(' ', '', s)
        s1 = re.sub('\n', 'e', s1)
        sl = list(s1)
        for a in range(len(sl)):
            if sl[a].encode('euc-kr') >= b'\xca\xa1' and \
               sl[a].encode('euc-kr') <= b'\xfd\xfe': sl[a] = 'h'
        s1 = ''.join(sl)
        return s1

    def _word2syllables(self, word):
        """
        word to syllables
        """
        syl_list = []

        dec = codecs.lookup('cp949').incrementaldecoder()
        w = self._normalize_func(dec.decode(word.encode('euc-kr')))
        for a in list(w):
            syl_list.append(a.encode('euc-kr').decode('euc-kr'))
        return syl_list

    def _get_cngram_syllable_wo_dic(self, word, min, max):
        """
        get syllables
        """
        word = word.replace('_', '')
        p_syl_list = self._word2syllables(word.upper())
        subword = []
        syl_list = p_syl_list[:]
        syl_list.insert(0, '<')
        syl_list.append('>')
        for a in range(len(syl_list)):
            for b in range(min, max+1):
                if a+b > len(syl_list): break
                x = syl_list[a:a+b]
                k = '_'.join(x)
                subword.append(k)
        return subword

    def _get_word_emb(self, w):
        """
        do word embedding
        """
        word = w.strip()
        assert len(word) > 0
        cng = self._get_cngram_syllable_wo_dic(word, self._subw_length_min,
                                               self._subw_length_max)
        lswi = [self._subw_dic[subw] for subw in cng if subw in self._subw_dic]
        if lswi == []: lswi = [self._subw_dic['UNK_SUBWORD']]
        d = np.sum(np.take(self._emb_np, lswi, axis=0), axis = 0)
        return d

    def get_sent_emb(self, s):
        """
        do sentence embedding
        """
        if s != '':
            s = s.strip().split()
            semb_tmp = []
            for a in s:
                semb_tmp.append(self._get_word_emb(a))
            avg = np.average(semb_tmp, axis=0)
        else:
            avg = np.zeros(self._emb_size)
        return avg