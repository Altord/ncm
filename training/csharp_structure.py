"""C# syntax-tree helpers for structure-aware code chunks and token losses."""

from __future__ import annotations

from functools import lru_cache


IDENTIFIER_NODES = {
    "identifier", "verbatim_identifier", "type_parameter",
}
LITERAL_NODES = {
    "integer_literal", "real_literal", "string_literal", "character_literal",
    "boolean_literal", "null_literal", "interpolated_string_expression",
}
COMMENT_NODES = {"comment", "documentation_comment"}
@lru_cache(maxsize=1)
def csharp_parser():
    try:
        from tree_sitter_language_pack import get_parser
    except ImportError as exc:
        raise RuntimeError(
            "AST-aware C# processing requires tree-sitter-language-pack; "
            "install requirements.txt in the project virtual environment"
        ) from exc
    return get_parser("csharp")


def _walk(node):
    stack = [node]
    while stack:
        current = stack.pop()
        yield current
        stack.extend(reversed(current.children))


def byte_to_char_offsets(text: str) -> list[int]:
    """Map each UTF-8 byte boundary to its containing Python character index."""
    result = [0]
    char_index = 0
    for char in text:
        encoded_len = len(char.encode("utf-8"))
        result.extend([char_index] * (encoded_len - 1))
        char_index += 1
        result.append(char_index)
    return result


def syntax_boundary_bytes(text: str) -> set[int]:
    """Return named-node ends, preferring AST token/construct boundaries."""
    tree = csharp_parser().parse(text.encode("utf-8"))
    return {
        node.end_byte for node in _walk(tree.root_node)
        if node.is_named and node.end_byte > 0
    }


def token_occurrence_weights(text: str, offsets: list[tuple[int, int]],
                             syntax_weight: float, identifier_weight: float,
                             default_weight: float = 1.0,
                             method_name_weight: float | None = None) -> list[float]:
    """Label tokenizer offsets using this occurrence's C# syntax tree.

    BPE pieces that overlap an identifier or literal receive identifier weight;
    other non-comment leaves receive syntax weight. Offsets are Python character
    offsets as returned by a fast Hugging Face tokenizer.
    """
    tree = csharp_parser().parse(text.encode("utf-8"))
    byte_to_char = byte_to_char_offsets(text)
    labels = bytearray(len(text))  # 0=other, 1=syntax, 2=identifier/literal
    for node in _walk(tree.root_node):
        if node.type in COMMENT_NODES:
            continue
        start = byte_to_char[min(node.start_byte, len(byte_to_char) - 1)]
        end = byte_to_char[min(node.end_byte, len(byte_to_char) - 1)]
        if end <= start:
            continue
        if node.type in IDENTIFIER_NODES or node.type in LITERAL_NODES:
            label = 2
        elif node.child_count == 0:
            label = 1
        else:
            continue
        for index in range(start, end):
            if label > labels[index]:
                labels[index] = label

    if method_name_weight is not None:
        for node in _walk(tree.root_node):
            if node.type != 'method_declaration':
                continue
            name = node.child_by_field_name('name')
            if name is None:
                continue
            start = byte_to_char[min(name.start_byte, len(byte_to_char) - 1)]
            end = byte_to_char[min(name.end_byte, len(byte_to_char) - 1)]
            for index in range(start, end):
                labels[index] = 3

    weights = []
    for start, end in offsets:
        if end <= start:
            weights.append(float(default_weight))
            continue
        label = max(labels[start:end], default=0)
        # Prefer identifier/literal weight when a BPE piece includes its leading space.
        weights.append(float(method_name_weight) if label == 3
                       else float(identifier_weight) if label == 2
                       else float(syntax_weight) if label == 1
                       else float(default_weight))
    return weights
