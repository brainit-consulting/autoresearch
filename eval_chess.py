"""
Standalone chess benchmark for the autoresearch chess PGN branch.

Measures:
- legal_move_rate from greedy next-move generation
- next_move_top1_accuracy from legal SAN candidate scoring
- next_move_topk_accuracy from legal SAN candidate scoring
"""

import argparse
import io
import random
import re
from dataclasses import dataclass

import chess
import chess.pgn
import torch
import torch.nn.functional as F

from prepare import MAX_SEQ_LEN, Tokenizer, _iter_chesspgn_texts, _resolve_dataset_name
from train import ASPECT_RATIO, DEPTH, HEAD_DIM, GPT, GPTConfig, WINDOW_PATTERN


SAN_RE = re.compile(r"^\s*(?:\d+\.(?:\.\.)?\s*)?([^\s]+)")


@dataclass
class EvalPosition:
    prompt: str
    target_san: str
    target_continuation: str
    board: chess.Board
    ply_prefix: int
    game_index: int


def _resolve_device(device_arg):
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return device_arg


def _build_model_config(vocab_size):
    base_dim = DEPTH * ASPECT_RATIO
    model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    num_heads = model_dim // HEAD_DIM
    return GPTConfig(
        sequence_len=MAX_SEQ_LEN,
        vocab_size=vocab_size,
        n_layer=DEPTH,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=model_dim,
        window_pattern=WINDOW_PATTERN,
        attention_backend="sdpa",
        use_activation_checkpointing=False,
        compute_dtype=torch.float32,
    )


def _load_model(checkpoint_path, tokenizer, device):
    model = GPT(_build_model_config(tokenizer.get_vocab_size()))
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def _format_prompt(moves, prefix_plies):
    board = chess.Board()
    parts = []
    for move in moves[:prefix_plies]:
        if board.turn == chess.WHITE:
            parts.append(f"{board.fullmove_number}.")
        parts.append(board.san(move))
        board.push(move)
    history = " ".join(parts)
    prompt = f"{history} " if history else ""
    target_move = moves[prefix_plies]
    target_san = board.san(target_move)
    if board.turn == chess.WHITE:
        target_continuation = f"{board.fullmove_number}. {target_san}"
    else:
        target_continuation = target_san
    return EvalPosition(
        prompt=prompt,
        target_san=target_san,
        target_continuation=target_continuation,
        board=board.copy(),
        ply_prefix=prefix_plies,
        game_index=-1,
    )


def _collect_positions(dataset, max_games, plies):
    positions = []
    parsed_games = 0
    for game_index, game_text in enumerate(_iter_chesspgn_texts("val", dataset_name=dataset)):
        if max_games is not None and parsed_games >= max_games:
            break
        game = chess.pgn.read_game(io.StringIO(game_text))
        if game is None:
            continue
        moves = list(game.mainline_moves())
        for prefix_plies in plies:
            if len(moves) <= prefix_plies:
                continue
            pos = _format_prompt(moves, prefix_plies)
            pos.game_index = game_index
            positions.append(pos)
        parsed_games += 1
    return positions, parsed_games


def _logprob_for_continuation(model, tokenizer, prompt_ids, continuation_text, device):
    continuation_ids = tokenizer.encode(continuation_text)
    if not continuation_ids:
        raise ValueError(f"Continuation produced no tokens: {continuation_text!r}")
    all_ids = prompt_ids + continuation_ids
    x = torch.tensor([all_ids[:-1]], dtype=torch.long, device=device)
    targets = torch.tensor(all_ids[1:], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(x)
        logprobs = F.log_softmax(logits, dim=-1)
    start = len(prompt_ids) - 1
    selected = logprobs[0, start:, :].gather(1, targets[start:].unsqueeze(1)).squeeze(1)
    return float(selected.sum().item())


def _greedy_generate_san(model, tokenizer, prompt_ids, board, device, max_new_tokens=24):
    ids = list(prompt_ids)
    for _ in range(max_new_tokens):
        x = torch.tensor([ids[-MAX_SEQ_LEN:]], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(x)
        next_id = int(torch.argmax(logits[0, -1]).item())
        ids.append(next_id)
        continuation = tokenizer.decode(ids[len(prompt_ids):])
        stripped = continuation.lstrip()
        if "\n" in stripped or "\r" in stripped:
            break
        match = SAN_RE.match(continuation)
        if not match:
            continue
        candidate = match.group(1)
        try:
            board.parse_san(candidate)
            return candidate
        except ValueError:
            continue
    continuation = tokenizer.decode(ids[len(prompt_ids):])
    match = SAN_RE.match(continuation)
    return match.group(1) if match else ""


def _candidate_continuation(board, san):
    if board.turn == chess.WHITE:
        return f"{board.fullmove_number}. {san}"
    return san


def _score_legal_moves(model, tokenizer, prompt_ids, board, device):
    scored = []
    for move in board.legal_moves:
        san = board.san(move)
        score = _logprob_for_continuation(model, tokenizer, prompt_ids, _candidate_continuation(board, san), device)
        scored.append((san, score))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored


def evaluate(args):
    dataset = _resolve_dataset_name(args.dataset)
    device = _resolve_device(args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = Tokenizer.from_directory(dataset=dataset)
    model = _load_model(args.checkpoint, tokenizer, device)
    positions, parsed_games = _collect_positions(dataset, args.max_games, args.plies)

    if not positions:
        raise RuntimeError("No evaluation positions found. Check dataset cache and --plies values.")

    legal_hits = 0
    top1_hits = 0
    topk_hits = 0
    total_candidates = 0
    examples = []

    for pos in positions:
        prompt_ids = tokenizer.encode(pos.prompt, prepend=tokenizer.get_bos_token_id())
        generated_san = _greedy_generate_san(model, tokenizer, prompt_ids, pos.board, device)
        is_legal = False
        if generated_san:
            try:
                pos.board.parse_san(generated_san)
                is_legal = True
            except ValueError:
                is_legal = False
        if is_legal:
            legal_hits += 1

        ranked = _score_legal_moves(model, tokenizer, prompt_ids, pos.board, device)
        total_candidates += len(ranked)
        ranked_sans = [san for san, _ in ranked]
        if ranked_sans and ranked_sans[0] == pos.target_san:
            top1_hits += 1
        if pos.target_san in ranked_sans[: args.top_k]:
            topk_hits += 1

        if len(examples) < args.greedy_samples:
            examples.append(
                {
                    "prompt": pos.prompt,
                    "generated": generated_san or "<empty>",
                    "generated_is_legal": is_legal,
                    "target": pos.target_san,
                    "top_candidates": ranked[:5],
                    "ply_prefix": pos.ply_prefix,
                    "game_index": pos.game_index,
                }
            )

    num_positions = len(positions)
    print("---")
    print(f"dataset:                 {dataset}")
    print(f"checkpoint:              {args.checkpoint}")
    print(f"device:                  {device}")
    print(f"games_scanned:           {parsed_games}")
    print(f"num_positions_evaluated: {num_positions}")
    print(f"plies:                   {','.join(str(p) for p in args.plies)}")
    print(f"legal_move_rate:         {legal_hits / num_positions:.4f}")
    print(f"next_move_top1_accuracy: {top1_hits / num_positions:.4f}")
    print(f"next_move_top{args.top_k}_accuracy: {topk_hits / num_positions:.4f}")
    print(f"avg_legal_candidates:    {total_candidates / num_positions:.2f}")

    print()
    print("Examples:")
    for idx, example in enumerate(examples, start=1):
        print(f"[{idx}] game={example['game_index']} prefix_plies={example['ply_prefix']}")
        print(f"prompt:          {example['prompt']}")
        print(f"generated_move:  {example['generated']}")
        print(f"generated_legal: {example['generated_is_legal']}")
        print(f"ground_truth:    {example['target']}")
        formatted = ", ".join(f"{san} ({score:.2f})" for san, score in example["top_candidates"])
        print(f"top_candidates:  {formatted}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Evaluate chess continuation quality for the autoresearch chess branch.")
    parser.add_argument("--checkpoint", required=True, help="Path to a saved model checkpoint (.pt state dict).")
    parser.add_argument("--dataset", choices=("chesspgn",), default="chesspgn", help="Dataset to evaluate.")
    parser.add_argument("--device", default="auto", help="Device to run on: auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--max-games", type=int, default=100, help="Maximum number of validation games to scan.")
    parser.add_argument("--plies", nargs="+", type=int, default=[8, 16], help="Evaluate prompts after these many plies.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic setup.")
    parser.add_argument("--greedy-samples", type=int, default=3, help="How many qualitative examples to print.")
    parser.add_argument("--top-k", type=int, default=3, help="Top-k accuracy cutoff for legal move ranking.")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
