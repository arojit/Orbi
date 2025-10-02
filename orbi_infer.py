
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orbi unified inference for four modes:
- storytelling
- tutor-math
- tutor-english (phonics / vocab)
- etiquette

Backends:
- Hugging Face Transformers (local model weights)
- llama.cpp server (Chat Completions API)
"""
import argparse
import json
import sys
from typing import List, Dict, Any, Optional

ORBI_SYSTEM = (
  "You are Orbi: a playful, caring mentor for children (ages 3–8). "
  "Always kid-safe, calm, and encouraging. Avoid scary or adult topics. "
  "Keep language simple. For stories, end with a cozy line. "
  "Use positive reinforcement. If a request is unsafe, cheerfully refuse and redirect."
)

def build_storytelling_messages(child_name, age_band, themes, values, tone="bedtime-soft", words=300, add_blocks=True):
    user = (
        f"Create a bedtime story for {child_name} (age band {age_band}) "
        f"about {', '.join(themes)}. Values: {', '.join(values)}. "
        f"Tone: {tone}. {words-50}-{words+50} words."
    )
    if add_blocks:
        user += " Include [illustrations] with 3–5 frame prompts and a [tool_calls] block with 2–4 lines."
    return [
        {"role": "system", "content": ORBI_SYSTEM},
        {"role": "user", "content": user},
    ]

def build_tutor_math_messages(prompt: str):
    return [
        {"role": "system", "content":
            "You are Orbi the tutor. Keep turns short, friendly, and age-appropriate. "
            "Ask one question at a time. Give gentle hints. Never shame; always encourage."
        },
        {"role": "user", "content": prompt},
    ]

def build_tutor_english_messages(prompt: str, phoneme_report: Optional[str] = None):
    u = prompt
    if phoneme_report:
        u += "\n" + phoneme_report
    return [
        {"role": "system", "content":
            "You are Orbi the tutor. Keep turns short, friendly, and age-appropriate. "
            "Ask one question at a time. Give gentle hints. Never shame; always encourage."
        },
        {"role": "user", "content": u},
    ]

def build_etiquette_messages(topic_prompt: str):
    return [
        {"role": "system", "content":
            "You are Orbi the etiquette coach. Teach manners in tiny lessons with positive reinforcement. "
            "Use kid-friendly examples and celebrate small wins."
        },
        {"role": "user", "content": topic_prompt},
    ]

def infer_hf(model_id, messages, max_new_tokens=1000, temperature=0.7, top_p=0.9):
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except Exception as e:
        raise RuntimeError("Transformers/Torch not available. pip install torch transformers") from e

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() or getattr(torch.backends, 'mps', None) else None,
        device_map="auto"
    ).eval()

    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p
    )
    text = tok.decode(out[0], skip_special_tokens=True)
    return text

def infer_llamacpp(server_url, messages, max_new_tokens=1000, temperature=0.7, top_p=0.9):
    import requests
    payload = {
        "model": "orbi",
        "messages": messages,
        "max_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p
    }
    r = requests.post(f"{server_url.rstrip('/')}/v1/chat/completions", json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        return json.dumps(data, ensure_ascii=False, indent=2)

def main():
    ap = argparse.ArgumentParser(description="Orbi unified inference")
    ap.add_argument("--backend", choices=["hf","llama.cpp"], default="hf")
    ap.add_argument("--model_id", default="./orbi-1b-merged")
    ap.add_argument("--server_url", default="http://127.0.0.1:8080")
    ap.add_argument("--mode", choices=["storytelling","tutor-math","tutor-english","etiquette"], required=True)

    ap.add_argument("--max_new_tokens", type=int, default=240)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)

    ap.add_argument("--child_name", default="Anaya")
    ap.add_argument("--age_band", choices=["3-4","5-6","7-8"], default="5-6")
    ap.add_argument("--themes", nargs="*", default=["friendship","sharing"])
    ap.add_argument("--values", nargs="*", default=["kindness","gratitude"])
    ap.add_argument("--tone", default="bedtime-soft")
    ap.add_argument("--words", type=int, default=300)

    ap.add_argument("--prompt", default=None)
    ap.add_argument("--phoneme_report", default=None)

    args = ap.parse_args()

    if args.mode == "storytelling":
        messages = build_storytelling_messages(
            child_name=args.child_name,
            age_band=args.age_band,
            themes=args.themes,
            values=args.values,
            tone=args.tone,
            words=args.words,
            add_blocks=True
        )
    elif args.mode == "tutor-math":
        prompt = args.prompt or "I got 9 + 7 wrong. Teach me the make-10 trick in 2 steps, then ask one question."
        messages = build_tutor_math_messages(prompt)
    elif args.mode == "tutor-english":
        prompt = args.prompt or "Help me say 'cat'."
        messages = build_tutor_english_messages(prompt, args.phoneme_report)
    elif args.mode == "etiquette":
        prompt = args.prompt or "Teach me how to say hello politely in 3 tiny steps and cheer for me."
        messages = build_etiquette_messages(prompt)
    else:
        print("Unknown mode", file=sys.stderr); sys.exit(1)

    if args.backend == "hf":
        text = infer_hf(
            model_id=args.model_id,
            messages=messages,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p
        )
    else:
        text = infer_llamacpp(
            server_url=args.server_url,
            messages=messages,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p
        )

    print(text)

if __name__ == "__main__":
    main()
