SPEECH_RATE_SLOW = 3.0
SPEECH_RATE_FAST = 5.5

AGITATION_HIGH = 0.65
AGITATION_MODERATE = 0.40


def determine_bipolar_phase(tachy: dict, emo: dict) -> dict:
    agitation = tachy.get("agitation_score", 0.0)
    speech_rate = tachy.get("speech_rate_syl_per_sec", 0.0)
    cnn_label = tachy.get("cnn_label", "normal")

    emotion = emo.get("emotion", "neutral")
    emo_conf = emo.get("confidence", 0.0)
    emo_probs = emo.get("probabilities", {})

    rate_fast = speech_rate > SPEECH_RATE_FAST
    rate_slow = speech_rate < SPEECH_RATE_SLOW
    rate_normal = not rate_fast and not rate_slow

    agit_high = agitation >= AGITATION_HIGH
    agit_mod = AGITATION_MODERATE <= agitation < AGITATION_HIGH

    phase_score_map = {
        "DEPRESSIF": -2,
        "DEPRESSIF_LEGER": -1,
        "EUTHYMIQUE": 0,
        "IRRITABILITE_RESIDUELLE": 0.5,
        "HYPOMANIAQUE": 1,
        "MANIAQUE": 2,
        "MIXTE": 99,
        "INDETERMINE": None
    }

    if emotion == "sad":
        if rate_slow or agitation < AGITATION_MODERATE or speech_rate < SPEECH_RATE_SLOW:
            phase = "DEPRESSIF"
            confidence = "élevée"
            explanation = "Affect triste avec ralentissement ou débit bas: phase dépressive probable."
        elif agit_high or rate_fast:
            phase = "HYPOMANIAQUE"
            confidence = "moyenne"
            explanation = "Affect triste avec activation vocale: profil mixte reclassé hypomaniaque dans cette taxonomie."
        else:
            phase = "DEPRESSIF"
            confidence = "moyenne"
            explanation = "Affect triste sans activation marquée: phase dépressive probable."

    elif emotion == "neutral":
        if rate_slow or agitation < AGITATION_MODERATE or speech_rate < SPEECH_RATE_SLOW:
            phase = "DEPRESSIF"
            confidence = "moyenne"
            explanation = "Voix neutre mais ralentie: profil dépressif possible."
        elif agit_high or rate_fast or speech_rate > SPEECH_RATE_FAST:
            phase = "MANIAQUE"
            confidence = "élevée"
            explanation = "Voix neutre avec forte activation: phase maniaque probable."
        elif agit_mod or speech_rate > SPEECH_RATE_SLOW:
            phase = "HYPOMANIAQUE"
            confidence = "moyenne"
            explanation = "Voix neutre avec accélération modérée: phase hypomaniaque probable."
        else:
            phase = "NEUTRE"
            confidence = "élevée"
            explanation = "Voix neutre, débit stable et agitation faible: phase neutre."

    elif emotion in {"happy", "angry"}:
        if agit_high or rate_fast or speech_rate > SPEECH_RATE_FAST or agitation >= AGITATION_HIGH:
            phase = "MANIAQUE"
            confidence = "élevée"
            explanation = "Émotion activée avec forte tachyphémie: phase maniaque probable."
        elif agit_mod or rate_normal or speech_rate >= SPEECH_RATE_SLOW:
            phase = "HYPOMANIAQUE"
            confidence = "moyenne"
            explanation = "Émotion activée avec activation modérée: phase hypomaniaque probable."
        else:
            phase = "NEUTRE"
            confidence = "faible"
            explanation = "Signal émotionnel activé mais insuffisant pour conclure."

    else:
        phase = "NEUTRE"
        confidence = "faible"
        explanation = "Signal ambigu ou émotion inconnue. Taxonomie ramenée à neutre."

    return {
        "phase": phase,
        "phase_score": phase_score_map.get(phase),
        "confidence": confidence,
        "explanation": explanation,
        "tachyphemia_label": cnn_label,
        "agitation_score": round(agitation, 4),
        "speech_rate_syl_s": round(speech_rate, 2),
        "emotion": emotion,
        "emotion_confidence": round(emo_conf, 4),
        "emotion_probs": emo_probs
    }
