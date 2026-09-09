import os
from ollama import Client

BASE  = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
MODEL = "gemma4:12b"
client = Client(host=BASE, timeout=1800)

parts, think, phase = [], [], None
for ch in client.generate(
    model=MODEL,
    prompt="Sta je python dekorator i kako se koristi?",
    stream=True,                                  # <-- ključno
    keep_alive=-1,
    options={"temperature": 0.2, "num_predict": 2048},   # <-- 128 je bilo premalo
):
    if ch.thinking:
        if phase != "t": print("\n[think] ", end="", flush=True); phase = "t"
        think.append(ch.thinking); print(ch.thinking, end="", flush=True)
    if ch.response:
        if phase != "r": print("\n[answer] ", end="", flush=True); phase = "r"
        parts.append(ch.response); print(ch.response, end="", flush=True)
    if ch.done:
        print(f"\n\n-- done_reason: {ch.done_reason} | eval: {ch.eval_count}")

answer = "".join(parts)
print("\nUKUPNO:", len(answer), "znakova |", len("".join(think)), "znakova thinking")