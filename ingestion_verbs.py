# external imports
import autocorrect
import requests
import json
import time
import spacy
import sys
import wn
from autocorrect import Speller
# local imports
import mongo_interface
from constants import *


def get_sketchengine_wsketch_rw(lemma: str, corpus: str, de: bool):
    lemma = lemma.replace('_', ' ')
    params = {
        'corpname': corpus,
        'format': 'json',
        'lemma': lemma,
        'lpos': '-'+SE_VERB,
        'maxitems': '100'  # cannot raise more than 100
    }
    url = SE_URL + '/wsketch?'
    response = requests.get(url, params=params, auth=(SE_USERNAME, SE_KEY))
    # time.sleep(5)
    print(response.request.url)
    # print(response.text)
    try:
        gramrels_unfiltered = json.loads(response.text)['Gramrels']
    except KeyError as e:
        print(e)
        gramrels_unfiltered = []
    genobj_scores, datobj_scores, accobj_scores, obj_scores, subj_scores, adjmod_scores = dict(), dict(), dict(), dict(), dict(), dict()
    for rel in gramrels_unfiltered:
        if rel['name'] == 'objects of \"%w\"':
            try:
                # nwords = len(rel['Words'])
                for i, wordinfo in enumerate(rel['Words']):
                    pos = wordinfo['lempos'][-1]
                    word = wordinfo['word'].lower()
                    if pos == SE_NOUN and word not in obj_scores:
                        # obj_scores[word] = round((nwords - i) / nwords, 3)    # relative score
                        obj_scores[word] = round(wordinfo['score'] / 14, 3)     # actual score
            except KeyError as e:
                print(rel['name'], e)
        elif rel['name'] == 'genitive objects of \"%w\"':
            try:
                # nwords = len(rel['Words'])
                for i, wordinfo in enumerate(rel['Words']):
                    pos = wordinfo['lempos'][-1]
                    word = wordinfo['word'].lower().capitalize()
                    if pos == SE_NOUN and word not in genobj_scores:
                        # genobj_scores[word] = round((nwords - i) / nwords, 3)    # relative score
                        genobj_scores[word] = round(wordinfo['score'] / 14, 3)     # actual score
            except KeyError as e:
                print(rel['name'], e)
        elif rel['name'] == 'dative objects of \"%w\"':
            try:
                # nwords = len(rel['Words'])
                for i, wordinfo in enumerate(rel['Words']):
                    pos = wordinfo['lempos'][-1]
                    word = wordinfo['word'].lower().capitalize()
                    if pos == SE_NOUN and word not in datobj_scores:
                        # datobj_scores[word] = round((nwords - i) / nwords, 3)    # relative score
                        datobj_scores[word] = round(wordinfo['score'] / 14, 3)     # actual score
            except KeyError as e:
                print(rel['name'], e)
        elif rel['name'] == 'accusative objects of \"%w\"':
            try:
                # nwords = len(rel['Words'])
                for i, wordinfo in enumerate(rel['Words']):
                    pos = wordinfo['lempos'][-1]
                    word = wordinfo['word'].lower().capitalize()
                    if pos == SE_NOUN and word not in accobj_scores:
                        # accobj_scores[word] = round((nwords - i) / nwords, 3)    # relative score
                        accobj_scores[word] = round(wordinfo['score'] / 14, 3)     # actual score
            except KeyError as e:
                print(rel['name'], e)
        elif rel['name'] == 'subjects of \"%w\"':
            try:
                # nwords = len(rel['Words'])
                for i, wordinfo in enumerate(rel['Words']):
                    pos = wordinfo['lempos'][-1]
                    word = wordinfo['word'].lower()
                    if de:
                        word = word.capitalize()
                    if pos == SE_NOUN and word not in subj_scores:
                        # subj_scores[word] = round((nwords - i) / nwords, 3)    # relative score
                        subj_scores[word] = round(wordinfo['score'] / 14, 3)     # actual score
            except KeyError as e:
                print(rel['name'], e)
        elif rel['name'] == 'complements of \"%w\"' or rel['name'] == 'adjectives after \"%w\"':
            try:
                # nwords = len(rel['Words'])
                for i, wordinfo in enumerate(rel['Words']):
                    pos = wordinfo['lempos'][-1]
                    word = wordinfo['word'].lower()
                    if pos == SE_ADJECTIVE and word not in adjmod_scores:
                        # adjmod_scores[word] = round((nwords - i) / nwords, 3)    # relative score
                        adjmod_scores[word] = round(wordinfo['score'] / 14, 3)  # actual score
            except KeyError as e:
                print(rel['name'], e)
        elif rel['name'] == 'modifiers of \"%w\"':
            try:
                # nwords = len(rel['Words'])
                for i, wordinfo in enumerate(rel['Words']):
                    pos = wordinfo['lempos'][-1]
                    word = wordinfo['word'].lower()
                    if word not in adjmod_scores and (pos == SE_ADJECTIVE or de):
                        # adjmod_scores[word] = round((nwords - i) / nwords, 3)    # relative score
                        adjmod_scores[word] = round(wordinfo['score'] / 14, 3)  # actual score
            except KeyError as e:
                print(rel['name'], e)
    if de:
        return adjmod_scores, genobj_scores, datobj_scores, accobj_scores, subj_scores
    else:
        return adjmod_scores, obj_scores, subj_scores


def merge_scores_len(*scoredicts):
    idxs_lens_scoredicts = sorted([(i, len(sd.keys())) for i, sd in enumerate(scoredicts)], key=lambda x: x[1])
    # print(idxs_lens_scoredicts)
    merged_scores = dict()
    for i, lensd in idxs_lens_scoredicts:
        for k, v in scoredicts[i].items():
            if k not in merged_scores:
                merged_scores[k] = v
            else:
                scores_ms = {origin: score for (score, origin) in merged_scores[k]}
                scores_sd = {origin: score for (score, origin) in v}
                new_v = set()
                for origin, score in scores_sd.items():
                    if origin not in scores_ms:
                        new_v.add((score, origin))
                for origin, score in scores_ms.items():
                    new_v.add((score, origin))
                merged_scores[k] = new_v
    return merged_scores


def merge_scores_keepsource(dict1: dict, dict2: dict, lemma: str):
    merged_scores = dict()
    for k, v1 in dict1.items():
        if k in dict2:
            merged_scores[k] = v1 | {(dict2[k], lemma)}
        else:
            merged_scores[k] = v1
    for k, v2 in dict2.items():
        if k not in dict1:
            merged_scores[k] = {(v2, lemma)}
    return merged_scores


def spacy_clean_stopwords(multiword: str, language: str):
    multiword = multiword.replace('_', ' ')
    corpus = SPACY_CORPUS_EN if language == EN else (
        SPACY_CORPUS_IT if language == IT else (SPACY_CORPUS_DE if language == DE else None))
    nlp = spacy.load(corpus, disable=['parser', 'ner'])
    clean_multiword = ''
    for token in nlp(multiword):
        if token.is_alpha and not token.is_stop:
            clean_multiword += token.lemma_ + ' '
    return clean_multiword[:-1]


# TODO: correggi tutti i dati prodotti sulla collection uncapitalizzali e correggi la funzione perchè non salvi roba capitalizzata?
def translate_mongo_panlex(google_sl: str, google_tl: str, q: str, gt_pos: str, mongo: bool, collection, collection_panlex, only_panlex: bool):
    q = q.lower()
    translations = None
    withmongo = False
    if not only_panlex:
        translations = mongo_interface.get_transl(collection, google_sl, google_tl, q, gt_pos) if mongo else None
    if not translations:
        sl = PANLEX_EN if google_sl == GT_EN else (PANLEX_IT if google_sl == GT_IT else PANLEX_DE)
        tl = PANLEX_EN if google_tl == GT_EN else (PANLEX_IT if google_tl == GT_IT else PANLEX_DE)
        if sl == PANLEX_DE and gt_pos == GT_NOUN:
            q = q.capitalize()
        translations = mongo_interface.get_transl(collection_panlex, sl, tl, q, None) if mongo else None
        if not translations:
            url = PANLEX_URL
            params_sl = {'uid': sl,
                         'txt': q}
            time.sleep(PANLEX_API_SLEEP_TIME)  # API users are requested not to perform more than 2 queries per second.
            response = requests.get(url, params=params_sl)
            panlex_id = json.loads(response.text)['result'][0]['id'] if len(json.loads(response.text)['result']) > 0 else None
            params_tl = {'uid': tl,
                         'trans_expr': panlex_id,
                         'include': 'trans_quality'}
            response = requests.get(url, params=params_tl) if panlex_id else None
            translations = [el[0].lower() for el in sorted([(el['txt'], el['trans_quality']) for el in json.loads(response.text)['result']], key=lambda x: x[1], reverse=True)] if response else []
            if mongo:
                mongo_interface.set_transl(collection_panlex, sl, tl, q, None, translations)
                translations = mongo_interface.get_transl(collection_panlex, sl, tl, q, None)
        else:
            withmongo = True
    else:
        withmongo = True
    # print(translations)
    translations = [el.lower() for el in translations['transl']]
    return translations, withmongo


def tripleassociation(rw_en: set, rw_it: set, rw_de: set, gt_pos: str, mongo: bool = True):
    rw = []
    rw_weak = []
    collection = mongo_interface.connect(mongo_interface.COLLECTION_TRANSL) if mongo else None
    collection_panlex = mongo_interface.connect(mongo_interface.COLLECTION_TRANSL_PANLEX) if mongo else None
    mcount, pcount = 0, 0
    # ENGLISH -> ITALIAN (CREATES DOUBLE-ASSOCIATION)
    for lemma_en in rw_en:
        translations, withmongo = translate_mongo_panlex(GT_EN, GT_IT, lemma_en, gt_pos, mongo, collection, collection_panlex, only_panlex=False)
        if withmongo:
            mcount += 1
        else:
            pcount += 1
        for lemma_tl in translations:
            if lemma_tl in rw_it:
                rw.append([lemma_en, lemma_tl])
                break
    # ENGLISH -> GERMAN (CREATES TRIPLE-ASSOCIATION)
    for idx_double, double in enumerate(rw):
        lemma_en = double[0]
        translations, withmongo = translate_mongo_panlex(GT_EN, GT_DE, lemma_en, gt_pos, mongo, collection, collection_panlex, only_panlex=False)
        if withmongo:
            mcount += 1
        else:
            pcount += 1

        for lemma_tl in translations:
            if lemma_tl in rw_de:
                rw[idx_double].append(lemma_tl)
                break
    # FILTER ONLY TRIPLETS
    rw = list(filter(lambda x: len(x) == 3, rw))
    # ITALIAN -> GERMAN or GERMAN -> ITALIAN (MARKS AND COLLECT WEAK TRIPLETS)
    for idx_triple, rw_en_it_de in enumerate(rw):
        lemma_it = rw_en_it_de[1]
        lemma_de = rw_en_it_de[2]

        translations_de, withmongo = translate_mongo_panlex(GT_IT, GT_DE, lemma_it, gt_pos, mongo, collection, collection_panlex, only_panlex=False)
        if withmongo:
            mcount += 1
        else:
            pcount += 1
        translations_it, withmongo = translate_mongo_panlex(GT_DE, GT_IT, lemma_de, gt_pos, mongo, collection, collection_panlex, only_panlex=False)
        if withmongo:
            mcount += 1
        else:
            pcount += 1

        if lemma_de not in translations_de and lemma_it not in translations_it:
            # print(rw_en_it_de, "WEAK")
            rw_weak.append(rw_en_it_de)
    # CLEAN AND RETURN
    rw = [t for t in rw if t not in rw_weak]
    return rw, rw_weak, mcount, pcount


def doubleassociation(rw_1: set, rw_2: set, l_1: str, l_2: str, gt_pos: str, mongo: bool = True):
    rw = []
    rw_weak = []
    collection = mongo_interface.connect(mongo_interface.COLLECTION_TRANSL) if mongo else None
    collection_panlex = mongo_interface.connect(mongo_interface.COLLECTION_TRANSL_PANLEX) if mongo else None
    mcount, pcount = 0, 0
    # L1 -> L2 (CREATES DOUBLE-ASSOCIATION)
    for lemma_1 in rw_1:
        translations, withmongo = translate_mongo_panlex(l_1, l_2, lemma_1, gt_pos, mongo, collection, collection_panlex, only_panlex=False)
        if withmongo:
            mcount += 1
        else:
            pcount += 1
        for lemma_tl in translations:
            if lemma_tl in rw_2:
                rw.append([lemma_1, lemma_tl])
                break
    # L2 -> L1 (CHECKS IF DOUBLE-ASSOCIATION IS STRONG OR WEAK)
    for doublet in rw:
        lemma_1, lemma_2 = doublet[0], doublet[1]
        translations, withmongo = translate_mongo_panlex(l_2, l_1, lemma_2, gt_pos, mongo, collection, collection_panlex, only_panlex=False)
        if withmongo:
            mcount += 1
        else:
            pcount += 1
        is_strong = False
        for lemma_tl in translations:
            if lemma_tl == lemma_1:
                is_strong = True
                break
        if not is_strong:
            rw_weak.append(doublet)
    rw = [d for d in rw if d not in rw_weak]
    return rw, rw_weak, mcount, pcount


def get_babelnetfile_bnids_wn_wnids(triplets: list, lpos: str, word_bss_mapping: dict = None, mongo: bool = False):
    if not word_bss_mapping:
        with open(PATH_UTILS + FILE_WORDS_BSS_JSONMAPPING_V5_HIGH_ONLYISALPHA, 'r') as f:
            word_bss_mapping = json.load(f)
    wn.config.data_directory = PATH_WN_DATA
    collection = mongo_interface.connect(mongo_interface.COLLECTION_POLISEMY) if mongo else None
    bpos = BABEL_SYNSET_ADJECTIVE if lpos == ADJECTIVE else (BABEL_SYNSET_VERB if lpos == VERB else (BABEL_SYNSET_NOUN if lpos == NOUN else None))
    wpos = WN_ADJ if lpos == ADJECTIVE else (WN_VERB if lpos == VERB else (WN_NOUN if lpos == NOUN else None))
    matches = []
    for triplet in triplets:
        synsets_en = set([s.id[7:].replace('s', 'a') for s in wn.synsets(triplet[0][0], lang=WN_EN)])
        synsets_it = set([s.id[7:].replace('s', 'a') for s in wn.synsets(triplet[0][1], lang=WN_IT)])
        synsets_de_ode = set([s.id for s in wn.synsets(triplet[0][2], lang=WN_DE)])
        synsets_de = set()
        for synset in synsets_de_ode:
            transl_de_en = wn.synset(synset).translate(lang='en')
            if len(transl_de_en) > 1:
                raise Exception('More translations to EN synsets for this one!')
            elif len(transl_de_en) == 1:
                synsets_de.add(transl_de_en[0].id[7:].replace('s', 'a'))
        wnsyns = {'wn:'+s.replace('-', '') for s in synsets_en.intersection(*[synsets_it, synsets_de]) if s[-1] == wpos}
        babsyns = []
        for lemma in triplet[0]:
            try:
                babsyns.append(set([bs for bs in word_bss_mapping[lemma.lower()].split(' ') if bs[-1] == bpos]))
            except:
                babsyns.append(set())
        if mongo:
            triplet_compact = '_'.join(triplet[0])
            mongo_interface.set_polisemy(collection, triplet_compact, bpos, list(babsyns[0]), list(babsyns[1]), list(babsyns[2]))
        babsyns = babsyns[0].intersection(*babsyns[1:3])
        if len(babsyns) > 1:  # filter only wordnet-related babsyns (noun 82115, verb 95882, adj 114038, adv 117659)
            non_wordnet_bss = set(filter(lambda x: int(x[3:-1]) > 117659, babsyns))
            if not len(non_wordnet_bss) == len(babsyns):
                babsyns -= non_wordnet_bss
        matches.append((triplet, babsyns, wnsyns))
    return matches


def get_babelnetfile_bnids_wn_wnids_doublets(triplets: list, l_1_wn: str, l_2_wn_ifdehere: str, lpos: str, word_bss_mapping: dict = None):
    if not word_bss_mapping:
        with open(PATH_UTILS + FILE_WORDS_BSS_JSONMAPPING_V5_HIGH_ONLYISALPHA, 'r') as f:
            word_bss_mapping = json.load(f)
    wn.config.data_directory = PATH_WN_DATA
    bpos = BABEL_SYNSET_ADJECTIVE if lpos == ADJECTIVE else (BABEL_SYNSET_VERB if lpos == VERB else (BABEL_SYNSET_NOUN if lpos == NOUN else None))
    wpos = WN_ADJ if lpos == ADJECTIVE else (WN_VERB if lpos == VERB else (WN_NOUN if lpos == NOUN else None))
    matches = []
    for triplet in triplets:
        if l_1_wn == WN_EN:
            synsets_1 = set([s.id[7:].replace('s', 'a') for s in wn.synsets(triplet[0][0], lang=l_1_wn)])
        elif l_1_wn == WN_IT:
            synsets_1 = set([s.id[7:].replace('s', 'a') for s in wn.synsets(triplet[0][1], lang=l_1_wn)])
        else:
            raise Exception("Wrong l matching!")
        if l_2_wn_ifdehere == WN_DE:
            synsets_2_ode = set([s.id for s in wn.synsets(triplet[0][2], lang=l_2_wn_ifdehere)])
            synsets_2 = set()
            for synset in synsets_2_ode:
                transl_de_en = wn.synset(synset).translate(lang='en')
                if len(transl_de_en) > 1:
                    raise Exception('More translations to EN synsets for this one!')
                elif len(transl_de_en) == 1:
                    synsets_2.add(transl_de_en[0].id[7:].replace('s', 'a'))
        else:
            if l_2_wn_ifdehere == WN_EN:
                synsets_2 = set([s.id[7:].replace('s', 'a') for s in wn.synsets(triplet[0][0], lang=l_2_wn_ifdehere)])
            elif l_2_wn_ifdehere == WN_IT:
                synsets_2 = set([s.id[7:].replace('s', 'a') for s in wn.synsets(triplet[0][1], lang=l_2_wn_ifdehere)])
            else:
                raise Exception("Wrong l matching!")
        wnsyns = {'wn:' + s.replace('-', '') for s in synsets_1.intersection(synsets_2) if s[-1] == wpos}
        babsyns = []
        for lemma in triplet[0]:
            if not lemma == '':
                try:
                    babsyns.append(set([bs for bs in word_bss_mapping[lemma.lower()].split(' ') if bs[-1] == bpos]))
                except:
                    babsyns.append(set())
        babsyns = babsyns[0].intersection(babsyns[1])
        if len(babsyns) > 1:  # filter only wordnet-related babsyns (noun 82115, verb 95882, adj 114038, adv 117659)
            non_wordnet_bss = set(filter(lambda x: int(x[3:-1]) > 117659, babsyns))
            if not len(non_wordnet_bss) == len(babsyns):
                babsyns -= non_wordnet_bss
        matches.append((triplet, babsyns, wnsyns))
    return matches


def bad_typed_words_correction(triplets: list, gt_pos: str, spell: autocorrect.Speller, word_bss_mapping: dict, collection, collection_panlex):
    fixed, already_right, unfixable = 0, 0, 0
    triplets_fixed = []
    for i, triplet in enumerate(triplets):
        if triplet[3] == '' and triplet[4] == '':
            word = triplet[0][0].lower()
            candidates = [w[1] for w in sorted(spell.get_candidates(word), key=lambda x: x[0], reverse=True)]  # ordinati per likelihood
            found = False
            for candidate in candidates:
                if candidate == word:
                    found = True
                    already_right += 1
                    triplets_fixed.append(triplet)
            if not found:
                print()
                print("TRIPLET:", triplet)
                print("CANDIDATES:", candidates)
                translations_it_en, _ = translate_mongo_panlex(GT_IT, GT_EN, triplet[0][1], gt_pos, True, collection, collection_panlex, only_panlex=False)
                print("TRANSLATION IT-->EN:", translations_it_en)
                translations_de_en, _ = translate_mongo_panlex(GT_DE, GT_EN, triplet[0][2], gt_pos, True, collection, collection_panlex, only_panlex=False)
                print("TRANSLATION DE-->EN:", translations_de_en)
                if word in translations_de_en and word in translations_it_en:
                    print(word, "(unchanged, found correspondence)")
                    already_right += 1
                    triplets_fixed.append(triplet)
                else:
                    for candidate in candidates:
                        if not found:
                            if candidate in translations_de_en and candidate in translations_it_en:
                                print(word, '-->', candidate)
                                lpos = NOUN if gt_pos == GT_NOUN else (ADJECTIVE if gt_pos == GT_ADJECTIVE else VERB)
                                found = True
                                fixed += 1
                                already_present = False
                                for t_check in triplets[:i]:
                                    if t_check[0] == candidate:
                                        if t_check[1] == triplet[0][1] and t_check[2] == triplet[0][2]:
                                            already_present = True
                                            print("triplet already present!")
                                if not already_present:
                                    match = get_babelnetfile_bnids_wn_wnids([[candidate, triplet[0][1], triplet[0][2]]], lpos, word_bss_mapping, mongo=True)
                                    bss = ' '.join(match[0][1]) if match[0][1] else ''
                                    wss = ' '.join(match[0][2]) if match[0][2] else ''
                                    new_triplet = [(candidate, triplet[0][1], triplet[0][2]), triplet[1], triplet[2], bss, wss]
                                    triplets_fixed.append(new_triplet)
                                    print("triplet not present: matched -- ", new_triplet)
                    if not found:
                        unfixable += 1
        else:
            triplets_fixed.append(triplet)
    print("already_right:", already_right, "fixed:", fixed, "unfixable:", unfixable)
    return triplets_fixed


def ingest_data(lex_en: set, lex_it: set, lex_de: set, babelnet_id: str, wordnet_id: str, create_logfile: bool, verbmod_orig: str):
    if babelnet_id is None:
        babelnet_id = 'none'

    if verbmod_orig in [INTRA_1_NPV_BASIC, INDOBJ_2_NPVPP, INTRA_10_NPV]:
        verbmod = INTRA
    elif verbmod_orig in [COPADJ_3_NPVADJ, PREDADJ_8_NPVNPADJ]:
        verbmod = ADJ
    elif verbmod_orig in [TRA_5_NPVNP, COPNP_6_NPVNPATTR, TRA_9_NPVNPREC]:
        verbmod = TRA
    elif verbmod_orig in [DOBJ_7_NPVNPRECNP]:
        verbmod = DOBJ
    else:
        raise Exception("verbmod_orig not managed!")

    if create_logfile:
        filename = '_'.join([wordnet_id.replace(':', ''), babelnet_id.replace(':', ''), verbmod_orig, str(time.time()).replace('.', '-')]) + '.txt'
        stdout = sys.stdout
        sys.stdout = open(PATH_LOGS + filename, 'w')

    print("\n\nRUNNING:", lex_en, lex_it, lex_de, babelnet_id, wordnet_id)
    start_tot = time.time()

    # SKETCH ENGINE API:
    # GET RELATED WORDS OF LEMMAS (NOUN, ADJECTIVES, VERB) FROM DIFFERENT CORPORA
    print("\n\nSKETCH ENGINE API:")
    start = time.time()
    spacytime = 0
    # EN
    scores_adj_wsketch_c1_en, scores_obj_wsketch_c1_en, scores_subj_wsketch_c1_en = dict(), dict(), dict()
    scores_adj_wsketch_c2_en, scores_obj_wsketch_c2_en, scores_subj_wsketch_c2_en = dict(), dict(), dict()
    scores_adj_wsketch_c3_en, scores_obj_wsketch_c3_en, scores_subj_wsketch_c3_en = dict(), dict(), dict()
    for lemma_en in lex_en:
        if '_' in lemma_en:
            print("\nSPACY CLEANING ON "+lemma_en+":")
            start_spacy = time.time()
            lemma_en = spacy_clean_stopwords(lemma_en, EN)
            print(lemma_en)
            end_spacy = time.time()
            print("SPACY CLEANING TIME:", round(end_spacy - start_spacy), "seconds (", round((end_spacy - start_spacy) / 60), "minutes )\n")
            start -= (end_spacy - start_spacy)
            spacytime += (end_spacy - start_spacy)
        scores_adj_wsketch_c1_en_s, scores_obj_wsketch_c1_en_s, scores_subj_wsketch_c1_en_s = get_sketchengine_wsketch_rw(lemma_en, SE_CORPUS_TENTEN_EN, de=False)
        scores_adj_wsketch_c2_en_s, scores_obj_wsketch_c2_en_s, scores_subj_wsketch_c2_en_s = get_sketchengine_wsketch_rw(lemma_en, SE_CORPUS_TJSI_EN, de=False)
        scores_adj_wsketch_c3_en_s, scores_obj_wsketch_c3_en_s, scores_subj_wsketch_c3_en_s = get_sketchengine_wsketch_rw(lemma_en, SE_CORPUS_EURLEX_EN, de=False)
        if verbmod == ADJ:
            print("SE ADJECTIVES TENTEN EN '" + lemma_en + "':", len(scores_adj_wsketch_c1_en_s.keys()), scores_adj_wsketch_c1_en_s)
            print("SE ADJECTIVES TJSI EN '" + lemma_en + "':",  len(scores_adj_wsketch_c2_en_s.keys()), scores_adj_wsketch_c2_en_s)
            print("SE ADJECTIVES EURLEX EN '" + lemma_en + "':", len(scores_adj_wsketch_c3_en_s.keys()), scores_adj_wsketch_c3_en_s)
            scores_adj_wsketch_c1_en = merge_scores_keepsource(scores_adj_wsketch_c1_en, scores_adj_wsketch_c1_en_s, lemma_en)
            scores_adj_wsketch_c2_en = merge_scores_keepsource(scores_adj_wsketch_c2_en, scores_adj_wsketch_c2_en_s, lemma_en)
            scores_adj_wsketch_c3_en = merge_scores_keepsource(scores_adj_wsketch_c3_en, scores_adj_wsketch_c3_en_s, lemma_en)
        if verbmod in [ADJ, TRA, DOBJ] and not verbmod_orig == COPADJ_3_NPVADJ:
            print("SE OBJECTS TENTEN EN '"+lemma_en+"':", len(scores_obj_wsketch_c1_en_s.keys()), scores_obj_wsketch_c1_en_s)
            print("SE OBJECTS TJSI EN '"+lemma_en+"':", len(scores_obj_wsketch_c2_en_s.keys()), scores_obj_wsketch_c2_en_s)
            print("SE OBJECTS EURLEX EN '"+lemma_en+"':", len(scores_obj_wsketch_c3_en_s.keys()), scores_obj_wsketch_c3_en_s)
            scores_obj_wsketch_c1_en = merge_scores_keepsource(scores_obj_wsketch_c1_en, scores_obj_wsketch_c1_en_s, lemma_en)
            scores_obj_wsketch_c2_en = merge_scores_keepsource(scores_obj_wsketch_c2_en, scores_obj_wsketch_c2_en_s, lemma_en)
            scores_obj_wsketch_c3_en = merge_scores_keepsource(scores_obj_wsketch_c3_en, scores_obj_wsketch_c3_en_s, lemma_en)
        print("SE SUBJECTS WSKETCH TENTEN EN '"+lemma_en+"':", len(scores_subj_wsketch_c1_en_s.keys()), scores_subj_wsketch_c1_en_s)
        print("SE SUBJECTS WSKETCH TJSI EN '"+lemma_en+"':", len(scores_subj_wsketch_c2_en_s.keys()), scores_subj_wsketch_c2_en_s)
        print("SE SUBJECTS WSKETCH EURLEX EN '"+lemma_en+"':", len(scores_subj_wsketch_c3_en_s.keys()), scores_subj_wsketch_c3_en_s)
        scores_subj_wsketch_c1_en = merge_scores_keepsource(scores_subj_wsketch_c1_en, scores_subj_wsketch_c1_en_s, lemma_en)
        scores_subj_wsketch_c2_en = merge_scores_keepsource(scores_subj_wsketch_c2_en, scores_subj_wsketch_c2_en_s, lemma_en)
        scores_subj_wsketch_c3_en = merge_scores_keepsource(scores_subj_wsketch_c3_en, scores_subj_wsketch_c3_en_s, lemma_en)
    # IT
    scores_adj_wsketch_c1_it, scores_obj_wsketch_c1_it, scores_subj_wsketch_c1_it = dict(), dict(), dict()
    scores_adj_wsketch_c2_it, scores_obj_wsketch_c2_it, scores_subj_wsketch_c2_it = dict(), dict(), dict()
    scores_adj_wsketch_c3_it, scores_obj_wsketch_c3_it, scores_subj_wsketch_c3_it = dict(), dict(), dict()
    for lemma_it in lex_it:
        if '_' in lemma_it:
            print("\nSPACY CLEANING ON "+lemma_it+":")
            start_spacy = time.time()
            lemma_it = spacy_clean_stopwords(lemma_it, IT)
            print(lemma_it)
            end_spacy = time.time()
            print("SPACY CLEANING TIME:", round(end_spacy - start_spacy), "seconds (", round((end_spacy - start_spacy) / 60), "minutes )\n")
            start -= (end_spacy - start_spacy)
            spacytime += (end_spacy - start_spacy)
        scores_adj_wsketch_c1_it_s, scores_obj_wsketch_c1_it_s, scores_subj_wsketch_c1_it_s = get_sketchengine_wsketch_rw(lemma_it, SE_CORPUS_TENTEN_IT, de=False)
        scores_adj_wsketch_c2_it_s, scores_obj_wsketch_c2_it_s, scores_subj_wsketch_c2_it_s = get_sketchengine_wsketch_rw(lemma_it, SE_CORPUS_TJSI_IT, de=False)
        scores_adj_wsketch_c3_it_s, scores_obj_wsketch_c3_it_s, scores_subj_wsketch_c3_it_s = get_sketchengine_wsketch_rw(lemma_it, SE_CORPUS_EURLEX_IT, de=False)
        if verbmod == ADJ:
            print("SE ADJECTIVES TENTEN IT '" + lemma_it + "':", len(scores_adj_wsketch_c1_it_s.keys()), scores_adj_wsketch_c1_it_s)
            print("SE ADJECTIVES TJSI IT '" + lemma_it + "':",  len(scores_adj_wsketch_c2_it_s.keys()), scores_adj_wsketch_c2_it_s)
            print("SE ADJECTIVES EURLEX IT '" + lemma_it + "':", len(scores_adj_wsketch_c3_it_s.keys()), scores_adj_wsketch_c3_it_s)
            scores_adj_wsketch_c1_it = merge_scores_keepsource(scores_adj_wsketch_c1_it, scores_adj_wsketch_c1_it_s, lemma_it)
            scores_adj_wsketch_c2_it = merge_scores_keepsource(scores_adj_wsketch_c2_it, scores_adj_wsketch_c2_it_s, lemma_it)
            scores_adj_wsketch_c3_it = merge_scores_keepsource(scores_adj_wsketch_c3_it, scores_adj_wsketch_c3_it_s, lemma_it)
        if verbmod in [ADJ, TRA, DOBJ] and not (verbmod_orig == COPADJ_3_NPVADJ or verbmod_orig == TRA_9_NPVNPREC):
            print("SE OBJECTS TENTEN IT '"+lemma_it+"':", len(scores_obj_wsketch_c1_it_s.keys()), scores_obj_wsketch_c1_it_s)
            print("SE OBJECTS TJSI IT '"+lemma_it+"':", len(scores_obj_wsketch_c2_it_s.keys()), scores_obj_wsketch_c2_it_s)
            print("SE OBJECTS EURLEX IT '"+lemma_it+"':", len(scores_obj_wsketch_c3_it_s.keys()), scores_obj_wsketch_c3_it_s)
            scores_obj_wsketch_c1_it = merge_scores_keepsource(scores_obj_wsketch_c1_it, scores_obj_wsketch_c1_it_s, lemma_it)
            scores_obj_wsketch_c2_it = merge_scores_keepsource(scores_obj_wsketch_c2_it, scores_obj_wsketch_c2_it_s, lemma_it)
            scores_obj_wsketch_c3_it = merge_scores_keepsource(scores_obj_wsketch_c3_it, scores_obj_wsketch_c3_it_s, lemma_it)
        print("SE SUBJECTS WSKETCH TENTEN IT '"+lemma_it+"':", len(scores_subj_wsketch_c1_it_s.keys()), scores_subj_wsketch_c1_it_s)
        print("SE SUBJECTS WSKETCH TJSI IT '"+lemma_it+"':", len(scores_subj_wsketch_c2_it_s.keys()), scores_subj_wsketch_c2_it_s)
        print("SE SUBJECTS WSKETCH EURLEX IT '"+lemma_it+"':", len(scores_subj_wsketch_c3_it_s.keys()), scores_subj_wsketch_c3_it_s)
        scores_subj_wsketch_c1_it = merge_scores_keepsource(scores_subj_wsketch_c1_it, scores_subj_wsketch_c1_it_s, lemma_it)
        scores_subj_wsketch_c2_it = merge_scores_keepsource(scores_subj_wsketch_c2_it, scores_subj_wsketch_c2_it_s, lemma_it)
        scores_subj_wsketch_c3_it = merge_scores_keepsource(scores_subj_wsketch_c3_it, scores_subj_wsketch_c3_it_s, lemma_it)
    # DE
    scores_adj_wsketch_c1_de, scores_genobj_wsketch_c1_de, scores_datobj_wsketch_c1_de, scores_accobj_wsketch_c1_de, scores_subj_wsketch_c1_de = dict(), dict(), dict(), dict(), dict()
    scores_adj_wsketch_c2_de, scores_genobj_wsketch_c2_de, scores_datobj_wsketch_c2_de, scores_accobj_wsketch_c2_de, scores_subj_wsketch_c2_de = dict(), dict(), dict(), dict(), dict()
    scores_adj_wsketch_c3_de, scores_genobj_wsketch_c3_de, scores_datobj_wsketch_c3_de, scores_accobj_wsketch_c3_de, scores_subj_wsketch_c3_de = dict(), dict(), dict(), dict(), dict()
    for lemma_de in lex_de:
        if '_' in lemma_de:
            print("\nSPACY CLEANING ON "+lemma_de+":")
            start_spacy = time.time()
            lemma_de = spacy_clean_stopwords(lemma_de, DE)
            print(lemma_de)
            end_spacy = time.time()
            print("SPACY CLEANING TIME:", round(end_spacy - start_spacy), "seconds (", round((end_spacy - start_spacy) / 60), "minutes )\n")
            start -= (end_spacy - start_spacy)
            spacytime += (end_spacy - start_spacy)
        scores_adj_wsketch_c1_de_s, scores_genobj_wsketch_c1_de_s, scores_datobj_wsketch_c1_de_s, scores_accobj_wsketch_c1_de_s, scores_subj_wsketch_c1_de_s = get_sketchengine_wsketch_rw(lemma_de, SE_CORPUS_TENTEN_DE, de=True)
        scores_adj_wsketch_c2_de_s, scores_genobj_wsketch_c2_de_s, scores_datobj_wsketch_c2_de_s, scores_accobj_wsketch_c2_de_s, scores_subj_wsketch_c2_de_s = get_sketchengine_wsketch_rw(lemma_de, SE_CORPUS_TJSI_DE, de=True)
        scores_adj_wsketch_c3_de_s, scores_genobj_wsketch_c3_de_s, scores_datobj_wsketch_c3_de_s, scores_accobj_wsketch_c3_de_s, scores_subj_wsketch_c3_de_s = get_sketchengine_wsketch_rw(lemma_de, SE_CORPUS_EURLEX_DE, de=True)
        if verbmod == ADJ:
            print("SE ADJECTIVES TENTEN DE '" + lemma_de + "':", len(scores_adj_wsketch_c1_de_s.keys()), scores_adj_wsketch_c1_de_s)
            print("SE ADJECTIVES TJSI DE '" + lemma_de + "':",  len(scores_adj_wsketch_c2_de_s.keys()), scores_adj_wsketch_c2_de_s)
            print("SE ADJECTIVES EURLEX DE '" + lemma_de + "':", len(scores_adj_wsketch_c3_de_s.keys()), scores_adj_wsketch_c3_de_s)
            scores_adj_wsketch_c1_de = merge_scores_keepsource(scores_adj_wsketch_c1_de, scores_adj_wsketch_c1_de_s, lemma_de)
            scores_adj_wsketch_c2_de = merge_scores_keepsource(scores_adj_wsketch_c2_de, scores_adj_wsketch_c2_de_s, lemma_de)
            scores_adj_wsketch_c3_de = merge_scores_keepsource(scores_adj_wsketch_c3_de, scores_adj_wsketch_c3_de_s, lemma_de)
        if verbmod in [ADJ, TRA, DOBJ] and not verbmod_orig == COPADJ_3_NPVADJ:
            if not verbmod_orig == TRA_9_NPVNPREC:
                if not verbmod == DOBJ:
                    print("SE GENITIVE OBJECTS TENTEN DE '"+lemma_de+"':", len(scores_genobj_wsketch_c1_de_s.keys()), scores_genobj_wsketch_c1_de_s)
                    print("SE GENITIVE OBJECTS TJSI DE '"+lemma_de+"':", len(scores_genobj_wsketch_c2_de_s.keys()), scores_genobj_wsketch_c2_de_s)
                    print("SE GENITIVE OBJECTS EURLEX DE '"+lemma_de+"':", len(scores_genobj_wsketch_c3_de_s.keys()), scores_genobj_wsketch_c3_de_s)
                print("SE ACCUSATIVE OBJECTS TENTEN DE '"+lemma_de+"':", len(scores_accobj_wsketch_c1_de_s.keys()), scores_accobj_wsketch_c1_de_s)
                print("SE ACCUSATIVE OBJECTS TJSI DE '"+lemma_de+"':", len(scores_accobj_wsketch_c2_de_s.keys()), scores_accobj_wsketch_c2_de_s)
                print("SE ACCUSATIVE OBJECTS EURLEX DE '"+lemma_de+"':", len(scores_accobj_wsketch_c3_de_s.keys()), scores_accobj_wsketch_c3_de_s)
            print("SE DATIVE OBJECTS TENTEN DE '"+lemma_de+"':", len(scores_datobj_wsketch_c1_de_s.keys()), scores_datobj_wsketch_c1_de_s)
            print("SE DATIVE OBJECTS TJSI DE '"+lemma_de+"':", len(scores_datobj_wsketch_c2_de_s.keys()), scores_datobj_wsketch_c2_de_s)
            print("SE DATIVE OBJECTS EURLEX DE '"+lemma_de+"':", len(scores_datobj_wsketch_c3_de_s.keys()), scores_datobj_wsketch_c3_de_s)
            scores_genobj_wsketch_c1_de, scores_datobj_wsketch_c1_de, scores_accobj_wsketch_c1_de = merge_scores_keepsource(scores_genobj_wsketch_c1_de, scores_genobj_wsketch_c1_de_s, lemma_de), merge_scores_keepsource(scores_datobj_wsketch_c1_de, scores_datobj_wsketch_c1_de_s, lemma_de), merge_scores_keepsource(scores_accobj_wsketch_c1_de, scores_accobj_wsketch_c1_de_s, lemma_de)
            scores_genobj_wsketch_c2_de, scores_datobj_wsketch_c2_de, scores_accobj_wsketch_c2_de = merge_scores_keepsource(scores_genobj_wsketch_c2_de, scores_genobj_wsketch_c2_de_s, lemma_de), merge_scores_keepsource(scores_datobj_wsketch_c2_de, scores_datobj_wsketch_c2_de_s, lemma_de), merge_scores_keepsource(scores_accobj_wsketch_c2_de, scores_accobj_wsketch_c2_de_s, lemma_de)
            scores_genobj_wsketch_c3_de, scores_datobj_wsketch_c3_de, scores_accobj_wsketch_c3_de = merge_scores_keepsource(scores_genobj_wsketch_c3_de, scores_genobj_wsketch_c3_de_s, lemma_de), merge_scores_keepsource(scores_datobj_wsketch_c3_de, scores_datobj_wsketch_c3_de_s, lemma_de), merge_scores_keepsource(scores_accobj_wsketch_c3_de, scores_accobj_wsketch_c3_de_s, lemma_de)
        print("SE SUBJECTS WSKETCH TENTEN DE '"+lemma_de+"':", len(scores_subj_wsketch_c1_de_s.keys()), scores_subj_wsketch_c1_de_s)
        print("SE SUBJECTS WSKETCH TJSI DE '"+lemma_de+"':", len(scores_subj_wsketch_c2_de_s.keys()), scores_subj_wsketch_c2_de_s)
        print("SE SUBJECTS WSKETCH EURLEX DE '"+lemma_de+"':", len(scores_subj_wsketch_c3_de_s.keys()), scores_subj_wsketch_c3_de_s)
        scores_subj_wsketch_c1_de = merge_scores_keepsource(scores_subj_wsketch_c1_de, scores_subj_wsketch_c1_de_s, lemma_de)
        scores_subj_wsketch_c2_de = merge_scores_keepsource(scores_subj_wsketch_c2_de, scores_subj_wsketch_c2_de_s, lemma_de)
        scores_subj_wsketch_c3_de = merge_scores_keepsource(scores_subj_wsketch_c3_de, scores_subj_wsketch_c3_de_s, lemma_de)
    end = time.time()
    print("\nSKETCH ENGINE TIME:", round(end - start), "seconds (", round((end - start) / 60), "minutes )")
    if spacytime > 0:
        print("SPACY TIME:", round(spacytime), "seconds (", round(spacytime/60), "minutes )")

    # MERGE SCORES AND OBTAIN RW VECTORS
    print("\n\nMERGE SCORES:")
    if verbmod == ADJ:
        adj_scores_wsketch_en = merge_scores_len(scores_adj_wsketch_c1_en, scores_adj_wsketch_c2_en, scores_adj_wsketch_c3_en)
        print("ADJ SCORES WSKETCH EN:", len(adj_scores_wsketch_en.keys()), adj_scores_wsketch_en)
        adj_scores_wsketch_it = merge_scores_len(scores_adj_wsketch_c1_it, scores_adj_wsketch_c2_it, scores_adj_wsketch_c3_it)
        print("ADJ SCORES WSKETCH IT:", len(adj_scores_wsketch_it.keys()), adj_scores_wsketch_it)
        adj_scores_wsketch_de = merge_scores_len(scores_adj_wsketch_c1_de, scores_adj_wsketch_c2_de, scores_adj_wsketch_c3_de)
        print("ADJ SCORES WSKETCH DE:", len(adj_scores_wsketch_de.keys()), adj_scores_wsketch_de)
        rw_adj_en, rw_adj_it, rw_adj_de = set(adj_scores_wsketch_en.keys()), set(adj_scores_wsketch_it.keys()), set(adj_scores_wsketch_de.keys())
        del scores_adj_wsketch_c1_en, scores_adj_wsketch_c2_en, scores_adj_wsketch_c3_en
        del scores_adj_wsketch_c1_it, scores_adj_wsketch_c2_it, scores_adj_wsketch_c3_it
        del scores_adj_wsketch_c1_de, scores_adj_wsketch_c2_de, scores_adj_wsketch_c3_de
    if verbmod in [ADJ, TRA, DOBJ] and not verbmod_orig == COPADJ_3_NPVADJ:
        obj_scores_wsketch_en = merge_scores_len(scores_obj_wsketch_c1_en, scores_obj_wsketch_c2_en, scores_obj_wsketch_c3_en)
        print("OBJ SCORES WSKETCH EN:", len(obj_scores_wsketch_en.keys()), obj_scores_wsketch_en)
        if not verbmod_orig == TRA_9_NPVNPREC:
            obj_scores_wsketch_it = merge_scores_len(scores_obj_wsketch_c1_it, scores_obj_wsketch_c2_it, scores_obj_wsketch_c3_it)
            print("OBJ SCORES WSKETCH IT:", len(obj_scores_wsketch_it.keys()), obj_scores_wsketch_it)
            if not verbmod == DOBJ:
                genobj_scores_wsketch_de = merge_scores_len(scores_genobj_wsketch_c1_de, scores_genobj_wsketch_c2_de, scores_genobj_wsketch_c3_de)
                print("GEN OBJ SCORES WSKETCH DE:", len(genobj_scores_wsketch_de.keys()), genobj_scores_wsketch_de)
            accobj_scores_wsketch_de = merge_scores_len(scores_accobj_wsketch_c1_de, scores_accobj_wsketch_c2_de, scores_accobj_wsketch_c3_de)
            print("ACC OBJ SCORES WSKETCH DE:", len(accobj_scores_wsketch_de.keys()), accobj_scores_wsketch_de)
        datobj_scores_wsketch_de = merge_scores_len(scores_datobj_wsketch_c1_de, scores_datobj_wsketch_c2_de, scores_datobj_wsketch_c3_de)
        print("DAT OBJ SCORES WSKETCH DE:", len(datobj_scores_wsketch_de.keys()), datobj_scores_wsketch_de)
        if verbmod == DOBJ:
            rw_obj_en, rw_obj_it, rw_obj_de_dat, rw_obj_de_acc = set(obj_scores_wsketch_en.keys()), set(obj_scores_wsketch_it.keys()), set(datobj_scores_wsketch_de.keys()), set(accobj_scores_wsketch_de.keys())
        elif verbmod_orig == TRA_9_NPVNPREC:
            rw_obj_en, rw_obj_de_dat = set(obj_scores_wsketch_en.keys()), set(datobj_scores_wsketch_de.keys())
        else:
            rw_obj_en, rw_obj_it, rw_obj_de = set(obj_scores_wsketch_en.keys()), set(obj_scores_wsketch_it.keys()), set(genobj_scores_wsketch_de.keys() | datobj_scores_wsketch_de.keys() | accobj_scores_wsketch_de.keys())
        del scores_obj_wsketch_c1_en, scores_obj_wsketch_c2_en, scores_obj_wsketch_c3_en
        del scores_obj_wsketch_c1_it, scores_obj_wsketch_c2_it, scores_obj_wsketch_c3_it
        del scores_genobj_wsketch_c1_de, scores_genobj_wsketch_c2_de, scores_genobj_wsketch_c3_de
        del scores_datobj_wsketch_c1_de, scores_datobj_wsketch_c2_de, scores_datobj_wsketch_c3_de
        del scores_accobj_wsketch_c1_de, scores_accobj_wsketch_c2_de, scores_accobj_wsketch_c3_de
    subj_scores_wsketch_en = merge_scores_len(scores_subj_wsketch_c1_en, scores_subj_wsketch_c2_en, scores_subj_wsketch_c3_en)
    print("SUBJ SCORES EN:", len(subj_scores_wsketch_en), subj_scores_wsketch_en)
    subj_scores_wsketch_it = merge_scores_len(scores_subj_wsketch_c1_it, scores_subj_wsketch_c2_it, scores_subj_wsketch_c3_it)
    print("SUBJ SCORES IT:", len(subj_scores_wsketch_it), subj_scores_wsketch_it)
    subj_scores_wsketch_de = merge_scores_len(scores_subj_wsketch_c1_de, scores_subj_wsketch_c2_de, scores_subj_wsketch_c3_de)
    print("SUBJ SCORES DE:", len(subj_scores_wsketch_de), subj_scores_wsketch_de)
    rw_subj_en, rw_subj_it, rw_subj_de = set(subj_scores_wsketch_en.keys()), set(subj_scores_wsketch_it.keys()), set(subj_scores_wsketch_de.keys())
    del scores_subj_wsketch_c1_en, scores_subj_wsketch_c2_en, scores_subj_wsketch_c3_en
    del scores_subj_wsketch_c1_it, scores_subj_wsketch_c2_it, scores_subj_wsketch_c3_it
    del scores_subj_wsketch_c1_de, scores_subj_wsketch_c2_de, scores_subj_wsketch_c3_de

    # PANLEX API / MONGO DB:
    # TRIPLETS CREATION (NOUN, MODIFIER(ADJECTIVE), VERB)
    print("\n\nPANLEX API / MONGO DB:")
    start = time.time()
    mcount, pcount = 0, 0
    if verbmod == ADJ:
        rw_adj_de = {rw.lower() for rw in rw_adj_de}
        triplets_adj, triplets_adj_weak, mcount_s, gcount_s = tripleassociation(rw_adj_en, rw_adj_it, rw_adj_de, GT_ADJECTIVE)
        mcount += mcount_s
        pcount += gcount_s
        print("ADJ TRIPLETS (STRONG):", len(triplets_adj), triplets_adj)
        print("ADJ TRIPLETS (WEAK):", len(triplets_adj_weak), triplets_adj_weak)
        end_adj = time.time()
    if verbmod in [ADJ, TRA] and not verbmod_orig in [COPADJ_3_NPVADJ, TRA_9_NPVNPREC]:
        rw_obj_de = {rw.lower() for rw in rw_obj_de}
        triplets_obj, triplets_obj_weak, mcount_s, gcount_s = tripleassociation(rw_obj_en, rw_obj_it, rw_obj_de, GT_NOUN)
        triplets_obj, triplets_obj_weak = [[triplet[0], triplet[1], triplet[2].capitalize()] for triplet in triplets_obj], [[triplet[0], triplet[1], triplet[2].capitalize()] for triplet in triplets_obj_weak]
        mcount += mcount_s
        pcount += gcount_s
        print("OBJ TRIPLETS (STRONG):", len(triplets_obj), triplets_obj)
        print("OBJ TRIPLETS (WEAK):", len(triplets_obj_weak), triplets_obj_weak)
        end_objs = time.time()
    elif verbmod == DOBJ or verbmod_orig == TRA_9_NPVNPREC:
        rw_obj_de_dat = {rw.lower() for rw in rw_obj_de_dat}
        doublets_dobj, doublets_dobj_weak, mcount_s, gcount_s = doubleassociation(rw_obj_en, rw_obj_de_dat, GT_EN, GT_DE, GT_NOUN)
        doublets_dobj, doublets_dobj_weak = [[doublet[0], doublet[1].capitalize()] for doublet in doublets_dobj], [[doublet[0], doublet[1].capitalize()] for doublet in doublets_dobj_weak]
        print("DOBJ/OBJ DOUBLETS (STRONG):", len(doublets_dobj), doublets_dobj)
        print("DOBJ/OBJ DOUBLETS (WEAK):", len(doublets_dobj_weak), doublets_dobj_weak)
        end_dobjs = time.time()
        if verbmod == DOBJ:
            rw_obj_de_acc = {rw.lower() for rw in rw_obj_de_acc}
            triplets_obj, triplets_obj_weak, mcount_s, gcount_s = tripleassociation(rw_obj_en, rw_obj_it, rw_obj_de_acc, GT_NOUN)
            triplets_obj, triplets_obj_weak = [[triplet[0], triplet[1], triplet[2].capitalize()] for triplet in triplets_obj], [[triplet[0], triplet[1], triplet[2].capitalize()] for triplet in triplets_obj_weak]
            print("OBJ TRIPLETS (STRONG):", len(triplets_obj), triplets_obj)
            print("OBJ TRIPLETS (WEAK):", len(triplets_obj_weak), triplets_obj_weak)
            end_objs = time.time()
    rw_subj_de = {rw.lower() for rw in rw_subj_de}
    triplets_subj, triplets_subj_weak, mcount_s, gcount_s = tripleassociation(rw_subj_en, rw_subj_it, rw_subj_de, GT_NOUN)
    triplets_subj, triplets_subj_weak = [[triplet[0], triplet[1], triplet[2].capitalize()] for triplet in triplets_subj], [[triplet[0], triplet[1], triplet[2].capitalize()] for triplet in triplets_subj_weak]
    mcount += mcount_s
    pcount += gcount_s
    print("SUBJ TRIPLETS (STRONG):", len(triplets_subj), triplets_subj)
    print("SUBJ TRIPLETS (WEAK):", len(triplets_subj_weak), triplets_subj_weak)
    end_subj = time.time()
    print("\nPANLEX / MONGO DB TIME:")
    if verbmod == ADJ:
        print("ADJS:", round(end_adj - start), "seconds (", round((end_adj - start) / 60), "minutes )")
    if verbmod in [ADJ, TRA] and not verbmod_orig in [COPADJ_3_NPVADJ, TRA_9_NPVNPREC]:
        if verbmod == ADJ:
            print("OBJS:", round(end_objs - end_adj), "seconds (", round((end_objs - end_adj) / 60), "minutes )")
        else:
            print("OBJS:", round(end_objs - start), "seconds (", round((end_objs - start) / 60), "minutes )")
        print("SUBJS:", round(end_subj - end_objs), "seconds (", round((end_subj - end_objs) / 60), "minutes )")
    elif verbmod == DOBJ or verbmod_orig == TRA_9_NPVNPREC:
        print("DOBJS/OBJS:", round(end_dobjs - start), "seconds (", round((end_dobjs - start) / 60), "minutes )")
        if verbmod == DOBJ:
            print("OBJS:", round(end_objs - end_dobjs), "seconds (", round((end_objs - end_dobjs) / 60), "minutes )")
            print("SUBJS:", round(end_subj - end_objs), "seconds (", round((end_subj - end_objs) / 60), "minutes )")
        else:
            print("SUBJS:", round(end_subj - end_dobjs), "seconds (", round((end_subj - end_dobjs) / 60), "minutes )")
    else:
        print("SUBJS:", round(end_subj - start), "seconds (", round((end_subj - start) / 60), "minutes )")
    print("REQUESTS AMOUNT:", mcount+pcount, "(MONGO DB: "+str(mcount)+")", "(PANLEX: "+str(pcount)+")")

    # SCORE CALCULATION, BABELNET FILE BS MATCHING, WN WS MATCHING:
    print("\n\nSCORE CALCULATION, BABELNET FILE BS MATCHING, WN WS MATCHING:")
    start = time.time()
    with open(PATH_UTILS + FILE_WORDS_BSS_JSONMAPPING_V5_HIGH_ONLYISALPHA, 'r') as f:
        word_bss_mapping = json.load(f)
    if verbmod == ADJ:
        triplets_adj_scores = []
        for triplet in triplets_adj:
            scores_origins_en = adj_scores_wsketch_en[triplet[0]]
            scores_origins_it = adj_scores_wsketch_it[triplet[1]]
            scores_origins_de = adj_scores_wsketch_de[triplet[2]]
            triplets_adj_scores.append([tuple(triplet), '0', ' '.join({word+' '+str(score) for (score, word) in scores_origins_en}), ' '.join({word+' '+str(score) for (score, word) in scores_origins_it}), ' '.join({word+' '+str(score) for (score, word) in scores_origins_de})])
        triplets_adj = [triplet + [' '.join(bss)] + [' '.join(wss)] for triplet, bss, wss in get_babelnetfile_bnids_wn_wnids(triplets_adj_scores, ADJECTIVE, word_bss_mapping, mongo=True) if not ' '.join(bss) == babelnet_id]
        print("ADJ TRIPLETS + SCORES AND MATCHINGS:", len(triplets_adj), triplets_adj)
    if verbmod in [ADJ, TRA, DOBJ] and not verbmod_orig in [COPADJ_3_NPVADJ, TRA_9_NPVNPREC]:
        triplets_obj_scores = []
        for triplet in triplets_obj:
            scores_origins_en = obj_scores_wsketch_en[triplet[0]]
            scores_origins_it = obj_scores_wsketch_it[triplet[1]]
            rw_de = triplet[2]
            if not verbmod == DOBJ:
                scores_origins_de_gen = {(el[0], el[1]) for el in genobj_scores_wsketch_de[rw_de]} if rw_de in genobj_scores_wsketch_de else {}
                scores_origins_de_dat = {(el[0], el[1]) for el in datobj_scores_wsketch_de[rw_de]} if rw_de in datobj_scores_wsketch_de else {}
                scores_origins_de_acc = {(el[0], el[1]) for el in accobj_scores_wsketch_de[rw_de]} if rw_de in accobj_scores_wsketch_de else {}
            if verbmod == DOBJ:
                triplets_obj_scores.append([tuple(triplet), '0', ' '.join({word+' '+str(score) for (score, word) in scores_origins_en}), ' '.join({word+' '+str(score) for (score, word) in scores_origins_it}), ' '.join({word+' '+str(score)+' (acc)' for (score, word) in scores_origins_de_acc})])
            else:
                triplets_obj_scores.append([tuple(triplet), '0', ' '.join({word+' '+str(score) for (score, word) in scores_origins_en}), ' '.join({word+' '+str(score) for (score, word) in scores_origins_it}), ' '.join({word+' '+str(score)+' (gen)' for (score, word) in scores_origins_de_gen})+' '+' '.join({word+' '+str(score)+' (dat)' for (score, word) in scores_origins_de_dat})+' '+' '.join({word+' '+str(score)+' (acc)' for (score, word) in scores_origins_de_acc})])
        triplets_obj = [triplet + [' '.join(bss)] + [' '.join(wss)] for triplet, bss, wss in get_babelnetfile_bnids_wn_wnids(triplets_obj_scores, NOUN, word_bss_mapping, mongo=True) if not ' '.join(bss) == babelnet_id]
        print("OBJ TRIPLETS WSKETCH + SCORES AND MATCHINGS:", len(triplets_obj), triplets_obj)
    if verbmod == DOBJ or verbmod_orig == TRA_9_NPVNPREC:
        triplets_dobj_scores = []
        for doublet in doublets_dobj:
            scores_origins_en = obj_scores_wsketch_en[doublet[0]]
            scores_origins_de_dat = {(el[0], el[1]) for el in datobj_scores_wsketch_de[doublet[1]]}
            triplets_dobj_scores.append([tuple([doublet[0], '', doublet[1]]), '0', ' '.join({word+' '+str(score) for (score, word) in scores_origins_en}), '', ' '.join({word+' '+str(score)+' (dat)' for (score, word) in scores_origins_de_dat})])
        triplets_dobj = [triplet + [' '.join(bss)] + [' '.join(wss)] for triplet, bss, wss in get_babelnetfile_bnids_wn_wnids_doublets(triplets_dobj_scores, WN_EN, WN_DE, NOUN, word_bss_mapping) if not ' '.join(bss) == babelnet_id]
        print("DOBJ/OBJ DOUBLETS WSKETCH + SCORES AND MATCHINGS:", len(triplets_dobj), triplets_dobj)
    triplets_subj_scores = []
    for triplet in triplets_subj:
        scores_origins_en = subj_scores_wsketch_en[triplet[0]]
        scores_origins_it = subj_scores_wsketch_it[triplet[1]]
        scores_origins_de = subj_scores_wsketch_de[triplet[2]]
        triplets_subj_scores.append([tuple(triplet), '0', ' '.join({word + ' ' + str(score) for (score, word) in scores_origins_en}), ' '.join({word + ' ' + str(score) for (score, word) in scores_origins_it}), ' '.join({word + ' ' + str(score) for (score, word) in scores_origins_de})])
    triplets_subj = [triplet + [' '.join(bss)] + [' '.join(wss)] for triplet, bss, wss in get_babelnetfile_bnids_wn_wnids(triplets_subj_scores, NOUN, word_bss_mapping, mongo=True) if not ' '.join(bss) == babelnet_id]
    print("SUBJ TRIPLETS + SCORES AND MATCHINGS:", len(triplets_subj), triplets_subj)
    end = time.time()
    print("\nBABELNET FILE BSS & WN WSS MAPPING TIME:", round(end - start), "seconds (", round((end - start) / 60), "minutes )")

    # BAD TYPED WORDS CHECK AND CORRECTION
    print("\n\nBAD TYPED WORDS CHECK AND CORRECTION")
    start = time.time()
    collection = mongo_interface.connect(mongo_interface.COLLECTION_TRANSL)
    collection_panlex = mongo_interface.connect(mongo_interface.COLLECTION_TRANSL_PANLEX)
    spell = Speller(lang='en')
    if verbmod == ADJ:
        triplets_adj = bad_typed_words_correction(triplets_adj, GT_ADJECTIVE, spell, word_bss_mapping, collection, collection_panlex)
    if verbmod in [ADJ, TRA, DOBJ] and not verbmod_orig in [COPADJ_3_NPVADJ, TRA_9_NPVNPREC]:
        triplets_obj = bad_typed_words_correction(triplets_obj, GT_NOUN, spell, word_bss_mapping, collection, collection_panlex)
    if verbmod == DOBJ or verbmod_orig == TRA_9_NPVNPREC:
        triplets_dobj = bad_typed_words_correction(triplets_dobj, GT_NOUN, spell, word_bss_mapping, collection, collection_panlex)
    triplets_subj = bad_typed_words_correction(triplets_subj, GT_NOUN, spell, word_bss_mapping, collection, collection_panlex)
    del word_bss_mapping
    end = time.time()
    print("\n\nBAD TYPED WORDS CHECK AND CORRECTION TIME:", round(end - start), "seconds (", round((end - start) / 60), "minutes )")

    end_tot = time.time()
    print("\n\nTOTAL TIME:", round(end_tot - start_tot), "seconds (", round((end_tot - start_tot) / 60), "minutes )")

    if create_logfile:
        sys.stdout.close()
        sys.stdout = stdout

    return dict(lex_en=lex_en,
                lex_it=lex_it,
                lex_de=lex_de,
                babelnet_id=babelnet_id,
                wordnet_id=wordnet_id,
                triplets_adj=sorted(triplets_adj, key=lambda x: (x[0][0], x[0][1]), reverse=True) if verbmod == ADJ else [],
                triplets_adj_weak=sorted(triplets_adj_weak, key=lambda x: (x[0][0], x[0][1])) if verbmod == ADJ else [],
                triplets_obj=sorted(triplets_obj, key=lambda x: (x[0][0], x[0][1]), reverse=True) if verbmod in [ADJ, TRA, DOBJ] and not verbmod_orig in [COPADJ_3_NPVADJ, TRA_9_NPVNPREC] else [],
                triplets_obj_weak=sorted(triplets_obj_weak, key=lambda x: (x[0][0], x[0][1])) if verbmod in [ADJ, TRA, DOBJ] and not verbmod_orig in [COPADJ_3_NPVADJ, TRA_9_NPVNPREC] else [],
                triplets_dobj=sorted(triplets_dobj, key=lambda x: (x[0][0], x[0][1]),reverse=True) if verbmod == DOBJ or verbmod_orig == TRA_9_NPVNPREC else [],
                triplets_dobj_weak=sorted([[el[0], '', el[1]] for el in doublets_dobj_weak], key=lambda x: (x[0][0], x[0][1])) if verbmod == DOBJ or verbmod_orig == TRA_9_NPVNPREC else [],
                triplets_subj=sorted(triplets_subj, key=lambda x: (x[0][0], x[0][1]), reverse=True),
                triplets_subj_weak=sorted(triplets_subj_weak, key=lambda x: (x[0][0], x[0][1])))

