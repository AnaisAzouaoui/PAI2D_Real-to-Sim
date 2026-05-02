"""
Script lance en sous-processus par ValidationWorker
Genesis tourne dans le main thread de ce processus 
"""
import sys
import os
import json

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(SRC_DIR, '..', '.env'))


def main():
    if len(sys.argv) < 4:
        print("Usage: run_validation.py <mode> <json_file> <prompt_or_image>", file=sys.stderr)
        sys.exit(1)

    mode       = sys.argv[1]
    json_file  = sys.argv[2]
    arg        = sys.argv[3]

    if mode == 'prompt':
        from boucle_validation.validation_prompt import boucle_vlm_prompt
        result = boucle_vlm_prompt(arg, json_file)
    elif mode == 'image':
        from boucle_validation.validation_img import boucle_vlm_img
        result = boucle_vlm_img(arg, json_file)
    else:
        print(f"Mode inconnu : {mode}", file=sys.stderr)
        sys.exit(1)

    if result:
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"[run_validation] scene corrigee sauvegardee -> {json_file}")
    else:
        print("[run_validation] aucun resultat retourne", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
