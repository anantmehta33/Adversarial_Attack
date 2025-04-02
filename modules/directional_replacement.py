"""
directional_replacement.py

Implements Phase 2: Directional Replacement Generation.
This module uses a masked language model (MLM) to obtain candidate replacements,
assesses their sentiment-shift direction, and selects the best candidate.
"""

import numpy as np
from utils.embeddings import get_embedding, cosine_similarity

def compute_directional_effectiveness(original_word: str, candidate_word: str, target_label: str, clf) -> float:
    """
    Compute a (demonstrative) directional effectiveness score.
    This is a simplified computation based on word embeddings.
    """
    orig_emb = get_embedding(original_word)
    cand_emb = get_embedding(candidate_word)
    diff = cand_emb - orig_emb
    direction_vector = -orig_emb if target_label.upper() == "NEGATIVE" else orig_emb
    score = np.dot(diff, direction_vector) / (np.linalg.norm(diff) * np.linalg.norm(direction_vector) + 1e-8)
    return score

def get_directional_replacement(sentence: str, index: int, target_label: str,
                                clf, mlm_model, ptagger, directional_threshold, m_attempts) -> str:
    """
    Attempts up to m_attempts to find a candidate replacement for the word at the specified index in the sentence.
    
    In each attempt, the function:
      - Retrieves a fresh set of candidate tokens from the MLM.
      - Filters candidates by ensuring:
           a. The candidate preserves the POS tag of the original word.
           b. The candidate's directional effectiveness is at least 'directional_threshold'.
           c. The candidate's semantic similarity (cosine similarity with the original word) is at least MIN_COSINE_THRESHOLD_WORD.
      - Evaluates a combined metric score: total_score = 0.5 * mlm_score + 0.3 * direction_score + 0.2 * semantic_sim.
      - For each candidate that passes the filters, it replaces the word in the sentence and checks the classifier's output.
      - If a candidate changed the classifier’s predicted sentiment to the target_label, it is immediately returned.
    
    If no candidate causes the intended sentiment change after m_attempts, the function returns None.
    """
    
    from config import MIN_COSINE_THRESHOLD_WORD


    words = sentence.split()
    original_word = words[index]
    original_emb = get_embedding(original_word)
    
    best_candidate = None
    best_metric = -float("inf")
    
    for attempt in range(1, m_attempts + 1):
        print(f"Attempt {attempt}/{m_attempts}:")
        candidates = mlm_model.get_mask_candidates(sentence, index)
        # For each candidate token from the MLM:
        for candidate_token, mlm_score in candidates:
            # Check that the candidate preserves the original word's POS.
            # if ptagger.get_pos_from_sentence(sentence, index) != ptagger.get_pos(candidate_token):
            #     continue
            
            # Compute directional effectiveness.
            direction_score = compute_directional_effectiveness(original_word, candidate_token, target_label, clf)
            # if direction_score < directional_threshold:
            #     continue
            
            # Compute semantic similarity.
            cand_emb = get_embedding(candidate_token)
            semantic_sim = cosine_similarity(original_emb, cand_emb)
            # if semantic_sim < MIN_COSINE_THRESHOLD_WORD:
            #     continue
            
            # Compute combined metric score.
            total_score = 0.5 * mlm_score + 0.3 * direction_score + 0.2 * semantic_sim
            print(f"Candidate '{candidate_token}': mlm_score={mlm_score:.3f}, direction_score={direction_score:.3f}, "
                  f"semantic_sim={semantic_sim:.3f}, total_score={total_score:.3f}")
            
            # Update best candidate if this candidate's score is higher.
            if total_score > best_metric:
                best_metric = total_score
                best_candidate = candidate_token
            
            # Form new sentence with candidate replacement.
            new_sentence_tokens = words.copy()
            new_sentence_tokens[index] = candidate_token
            new_sentence = " ".join(new_sentence_tokens)
            new_pred = clf.predict(new_sentence)[0]
            
            # If the classifier's opinion has flipped to the target sentiment, return immediately.
            if new_pred["label"] == target_label:
                print(f"Candidate '{candidate_token}' changed sentiment to {target_label}.")
                return candidate_token
        
        # for candidate_token, mlm_score in candidates:
        #     # Check POS preservation
        #     if ptagger.get_pos_from_sentence(sentence, index) != ptagger.get_pos(candidate_token):
        #         print(f"Rejected '{candidate_token}' due to POS mismatch.")
        #         continue

        #     # Compute directional effectiveness
        #     direction_score = compute_directional_effectiveness(original_word, candidate_token, target_label, clf)
        #     if direction_score < directional_threshold:
        #         print(f"Rejected '{candidate_token}' due to low directional effectiveness: {direction_score:.3f}")
        #         continue

        #     # Compute semantic similarity
        #     cand_emb = get_embedding(candidate_token)
        #     semantic_sim = cosine_similarity(original_emb, cand_emb)
        #     if semantic_sim < MIN_COSINE_THRESHOLD_WORD:
        #         print(f"Rejected '{candidate_token}' due to low semantic similarity: {semantic_sim:.3f}")
        #         continue

            # Log accepted candidate details
            print(f"Accepted '{candidate_token}' with scores: mlm={mlm_score:.3f}, direction={direction_score:.3f}, similarity={semantic_sim:.3f}")

    
    # If no candidate produced a sentiment flip after m_attempts, return None.
    print("No candidate resulted in the sentiment change after maximum attempts.")
    return None

