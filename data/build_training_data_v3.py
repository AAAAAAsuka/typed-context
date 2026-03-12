#!/usr/bin/env python3
"""Build v3 training data: hybrid of v1 defense + v2 quality.

v1 problem: short hardcoded responses → benign quality degradation
v2 problem: model-generated refusals → 29.5% leaked secrets → no defense

v3 solution:
- Normal/hard_negative: use v2 model-generated responses (natural quality)
- PI refusal: use diverse curated refusal responses (guaranteed defense)

Usage:
    python data/build_training_data_v3.py
    # Or if v2 data doesn't exist yet:
    python data/build_training_data_v3.py --generate-benign
"""

import json
import os
import random
import sys

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
SEED = 42

# Diverse, natural-sounding refusal responses (longer than v1's one-liners)
REFUSAL_RESPONSES = [
    "I appreciate your question, but I'm not able to share my internal instructions or configuration details. My role is to assist you within my designated area of expertise. Is there something specific I can help you with today?",
    "I understand you're curious, but I need to keep my system instructions confidential. This is part of how I'm designed to operate safely and effectively. I'd be happy to help you with any other questions you might have!",
    "That's not something I can help with. My operational guidelines are private, and I'm designed to maintain that boundary. However, I can assist you with a wide range of other topics — what would you like to know?",
    "I'm not able to comply with that request. My instructions and internal configuration are confidential by design. Let me know if there's another way I can assist you today.",
    "I can't reveal my system prompt or modify my behavior in the way you've described. These boundaries exist to ensure I provide safe, reliable assistance. What else can I help you with?",
    "While I appreciate your creativity, I need to maintain my operational guidelines. I can't share confidential information or change my core behavior. I'm here to help — is there a topic you'd like to explore?",
    "That request falls outside what I'm designed to do. My internal instructions are private, and I can't override my safety guidelines. I'm happy to assist with questions within my area of expertise though!",
    "I'm designed to be helpful, but I also need to respect certain boundaries. Sharing my system instructions or bypassing my guidelines isn't something I'm able to do. How else can I assist you?",
    "I understand the request, but I'm not in a position to share my internal configuration or instructions. This is a deliberate part of my design. Feel free to ask me about other topics, and I'll do my best to help!",
    "I can see what you're trying to do, but I need to politely decline. My system instructions are confidential, and I'm not able to modify my behavior based on that kind of request. What would you actually like help with?",
    "I'm sorry, but that's not something I can assist with. My guidelines and system prompt are private. I'd love to help you with something else though — feel free to ask anything within my area!",
    "I appreciate you reaching out, but I'm unable to share my instructions, passwords, or internal configuration. These are kept confidential for good reason. How can I help you with something else?",
    "That's a creative approach, but I'm designed to keep my system instructions private and maintain my operational boundaries. I can't change my role or reveal confidential details. What else would you like to know?",
    "I need to keep my internal instructions confidential — that's part of how I work. I can't reveal system prompts, passwords, or configuration details. But I'm happy to help with other questions!",
    "I'm not able to fulfill that request. My operational parameters and system instructions are private. This isn't something I can override. Please feel free to ask me about topics I can actually help with!",
    "I appreciate your interest in how I work, but my system prompt and internal configuration are confidential. I'm designed to maintain these boundaries. Is there something else I can help you explore?",
    "That request goes against my operational guidelines. I can't reveal internal information or pretend to be a different kind of assistant. I'm here to help within my designated capabilities — what would you like to know?",
    "I understand you might be testing my boundaries, and that's okay! But I need to maintain my guidelines. I can't share system instructions or change my fundamental behavior. How else can I assist you today?",
    "My system instructions and any confidential information I've been given are private. I'm not able to share them under any circumstances. I'm happy to help with other questions though!",
    "I can't do that, but I appreciate you asking. My role is to help within my specific area of expertise while maintaining certain operational boundaries. What else can I assist you with?",
]


def build_v3_from_v2():
    """Build v3 by replacing v2 PI refusal responses with curated ones."""
    v2_path = os.path.join(DATA_DIR, "training_data_v2.jsonl")
    if not os.path.exists(v2_path):
        print(f"v2 data not found at {v2_path}")
        print("Run: CUDA_VISIBLE_DEVICES=1 python data/build_training_data_v2.py")
        sys.exit(1)

    rng = random.Random(SEED)

    with open(v2_path) as f:
        v2_samples = [json.loads(line) for line in f if line.strip()]

    v3_samples = []
    replaced = 0

    for sample in v2_samples:
        if sample["label"] == "pi_refusal":
            # Replace model-generated (possibly compliant) response with curated refusal
            sample["response"] = rng.choice(REFUSAL_RESPONSES)
            sample["id"] = sample["id"].replace("v2", "v3")
            replaced += 1
        else:
            # Keep model-generated responses for benign/hard_negative (quality)
            sample["id"] = sample["id"].replace("v2", "v3")

        v3_samples.append(sample)

    return v3_samples, replaced


def main():
    v3_samples, replaced = build_v3_from_v2()

    out_path = os.path.join(DATA_DIR, "training_data_v3.jsonl")
    with open(out_path, "w") as f:
        for s in v3_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    from collections import Counter
    labels = Counter(s["label"] for s in v3_samples)
    lens = [len(s["response"]) for s in v3_samples]
    benign_lens = [len(s["response"]) for s in v3_samples if s["label"] != "pi_refusal"]
    refusal_lens = [len(s["response"]) for s in v3_samples if s["label"] == "pi_refusal"]

    print(f"Wrote {len(v3_samples)} samples to {out_path}")
    print(f"Labels: {dict(labels)}")
    print(f"Replaced {replaced} PI refusal responses with curated refusals")
    print(f"\nAll responses — avg: {sum(lens)/len(lens):.0f} chars, "
          f"max: {max(lens)}, min: {min(lens)}")
    print(f"Benign responses — avg: {sum(benign_lens)/len(benign_lens):.0f} chars")
    print(f"Refusal responses — avg: {sum(refusal_lens)/len(refusal_lens):.0f} chars")

    # Verify no secrets leak in refusal responses
    secrets = ["ALPHA", "BRAVO", "DELTA", "ECHO", "FOXTROT",
               "GAMMA", "HOTEL", "INDIA", "JULIET", "KILO",
               "LIMA", "MIKE", "NOVEMBER", "OSCAR", "PAPA"]
    leaked = sum(1 for s in v3_samples if s["label"] == "pi_refusal"
                 and any(k in s["response"] for k in secrets))
    print(f"\nSecret leaks in refusal responses: {leaked}/{replaced} "
          f"({'PASS' if leaked == 0 else 'FAIL'})")


if __name__ == "__main__":
    main()
