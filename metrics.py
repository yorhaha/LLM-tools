from nltk.translate.bleu_score import sentence_bleu

def compute_bleu_score(pred_list, gold_list):
    pred_tokenized = [list(jieba.cut(sentence)) for sentence in pred_list]
    gold_tokenized = [list(jieba.cut(sentence)) for sentence in gold_list]

    bleu_scores = [
        sentence_bleu([reference], candidate)
        for reference, candidate in zip(gold_tokenized, pred_tokenized)
    ]

    average_bleu = sum(bleu_scores) / len(bleu_scores)
    return average_bleu

from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("input/bge-small-zh-v1.5", device="cuda:0")

def select_best_match_by_embedding(question, candidates, cutoff=0.5):
    instruction = "为这个句子生成表示以用于检索相关文章："
    question = instruction + question
    question_embedding = model.encode(question, normalize_embeddings=True)
    candidate_embeddings = model.encode(candidates, normalize_embeddings=True)
    cos_scores = util.pytorch_cos_sim(question_embedding, candidate_embeddings)[0]

    # sort by scores
    sorted_idx = cos_scores.argsort().flip(0)
    cos_scores = cos_scores[sorted_idx]
    candidates = [candidates[i] for i in sorted_idx]

    # select scores > cutoff
    res = []
    for i in range(len(cos_scores)):
        if cos_scores[i] > cutoff:
            res.append(candidates[i])

    res = res[:10]

    return res
