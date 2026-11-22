"""Input and context guardrails: PII redaction and prompt-injection screening of retrieved text.

redact_pii(text)        replaces Aadhaar, PAN, Indian mobile numbers and e-mail addresses with placeholders; used on
                        every user message before it reaches the agent, and on every trace attribute
scan_injection(text)    names the injection patterns found in a retrieved chunk; the agent quarantines flagged
                        chunks so they never reach the LLM (Agent(injection_guard=True), the default)

Both are deterministic regex checks. They catch the common forms; the agent's rule critic and repair step are the
second layer (dates, findings and statuses come from tools, drafts are forced to sent=false, citations are limited to
retrieved sources), and a human reviewer is the last.
"""
from __future__ import annotations

import re

PII_PATTERNS = [
    ("AADHAAR", re.compile(r"(?<!\d)[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}(?!\d)")),
    ("PAN", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b", re.I)),
    ("PHONE", re.compile(r"(?<![\d-])(?:\+91[ -]?|0)?[6-9]\d{4}[ -]?\d{5}(?![\d-])")),
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")),
]

INJECTION_PATTERNS = [
    ("OVERRIDE", re.compile(r"\b(ignore|disregard|forget|override)\b[^.\n]{0,40}\b(previous|prior|above|earlier|all|"
                            r"your|system)\b[^.\n]{0,20}\b(instructions?|rules|prompts?|polic(y|ies)|guidelines)\b", re.I)),
    ("NEW_INSTRUCTIONS", re.compile(r"\b(new|updated|revised|real)\s+(system\s+)?instructions?\b", re.I)),
    ("ROLE_MARKER", re.compile(r"(?m)^\s*(system|assistant|developer)\s*:|<\|im_start\|>|\[/?INST\]|"
                               r"</?\s*(system|sources|instructions?|tool_results?)\s*>", re.I)),
    ("PERSONA", re.compile(r"\byou are now\b|\bpretend (to be|you are)\b|\bdeveloper mode\b|\bjailbreak\b|"
                           r"\bact as an? (unrestricted|different|new|uncensored)\b", re.I)),
    ("ADDRESSES_MODEL", re.compile(r"\b(AI|LLM|language model|chatbot|copilot|AI assistant|virtual assistant)s?\b"
                                   r"[^.\n]{0,50}\b(must|should|shall|will now|is (now )?(allowed|authori[sz]ed|permitted))\b")),
    ("OUTPUT_CONTROL", re.compile(r"\b(reply|respond|answer|output|print)\s+(only\s+)?(with|using)\s+(the\s+)?"
                                  r"(code|word|phrase|string|text)\b|\b(reply|respond|answer)\s+only\b", re.I)),
    ("ACTION_HIJACK", re.compile(r"[\"']?sent[\"']?\s*[:=]\s*true\b|\b(mark|set|flag)\b[^.\n]{0,20}\bdrafts?\b[^.\n]{0,20}"
                                 r"\bas sent\b|\b(send|e-?mail|forward)\b[^.\n]{0,40}\b(to|at)\b[^.\n]{0,20}@", re.I)),
    ("CONCEAL", re.compile(r"\b(do not|don't|never)\s+(tell|mention|reveal|inform|show)\b[^.\n]{0,20}\b(user|"
                           r"applicant|customer)\b", re.I)),
    ("FAKE_CITATION", re.compile(r"\bcite\b[^.\n]{0,30}\bSRC-[A-Z0-9-]+\b[^.\n]{0,30}\b(every|all|always)\b|"
                                 r"\balways cite\b", re.I)),
]


def redact_pii(text: str) -> tuple[str, list[str]]:
    """Returns the text with PII replaced by [TYPE REDACTED] and the list of PII types found."""
    found = []
    for name, pattern in PII_PATTERNS:
        text, n = pattern.subn(f"[{name} REDACTED]", text)
        if n:
            found.append(name)
    return text, found


def scan_injection(text: str) -> list[str]:
    return [name for name, pattern in INJECTION_PATTERNS if pattern.search(text or "")]
