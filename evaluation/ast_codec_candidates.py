"""Conservative C# AST error-node substitutions using supplied parameters."""
import re
from evaluation.codec_prompt_signature_copy import PROMPT_PREFIX, WRAPPER, parse_method
from training.csharp_structure import csharp_parser, _walk


def ast_candidates(prompt, text, limit=12):
    if not prompt.startswith(PROMPT_PREFIX) or limit < 1:
        return []
    signature = prompt[len(PROMPT_PREFIX):].strip()
    if parse_method(signature + ' {}', require_clean=True) is None:
        return []
    m = re.search(r'\(([^()]*)\)\s*$', signature)
    if not m:
        return []
    params = []
    for part in m.group(1).split(','):
        match = re.fullmatch(r'\s*(?:(?:ref|out|in|params)\s+)?[A-Za-z_]\w*(?:\[\])?\s+([A-Za-z_]\w*)\s*', part)
        if not match:
            return []
        params.append(match.group(1))
    parsed = parse_method(text)
    if not parsed or not params:
        return []
    wrapper = WRAPPER.encode('utf-8')
    raw = text.encode('utf-8')
    tree = csharp_parser().parse(wrapper + raw + b'\n}')
    result = []
    for node in _walk(tree.root_node):
        if node.type != 'identifier' or node.parent is None or node.parent.type != 'ERROR':
            continue
        ancestor = node.parent
        protected = False
        while ancestor is not None:
            kind = ancestor.type
            if ('string' in kind or 'comment' in kind or 'character_literal' in kind or
                kind in ('member_access_expression', 'argument_name', 'name_colon')):
                protected = True
                break
            ancestor = ancestor.parent
        if protected:
            continue
        start, end = node.start_byte-len(wrapper), node.end_byte-len(wrapper)
        if start < parsed['body'] or end > len(raw):
            continue
        word = raw[start:end].decode('utf-8')
        if word in params or not word.islower() or word.startswith('_'):
            continue
        # Do not touch member names, labels, or assignment targets.
        before, after = raw[parsed['body']:start].rstrip(), raw[end:].lstrip()
        if before.endswith(b'.') or after.startswith((b':', b'=', b'(')):
            continue
        for param in params:
            candidate = (raw[:start] + param.encode('utf-8') + raw[end:]).decode('utf-8')
            if candidate not in result:
                result.append(candidate)
                if len(result) >= limit:
                    return result
    return result
