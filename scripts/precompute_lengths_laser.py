#!/usr/bin/env python3
"""
Ultra-Fast Precompute Sequence Lengths

Core optimization: completely skip processor, directly calculate token count
- Image tokens: calculate directly based on Qwen2.5-VL resize logic
- Text tokens: use only tokenizer (not full processor)
- 10x+ speedup

Qwen2.5-VL image processing logic:
1. Resize while preserving aspect ratio within [min_pixels, max_pixels] range
2. Round height and width to multiples of 28
3. token count = (H // 28) * (W // 28)
"""

import argparse
import json
import math
import os
import struct
import multiprocessing
from tqdm import tqdm

# Global variables
_tokenizer = None
_image_folder = None
_min_pixels = None
_max_pixels = None
_patch_size = 28


def get_image_size_fast(filepath):
    """
    Fast read image dimensions, only read file header, no pixel decoding
    Supports JPEG, PNG, GIF, BMP, WebP
    """
    with open(filepath, 'rb') as f:
        head = f.read(32)

        # PNG
        if head[:8] == b'\x89PNG\r\n\x1a\n':
            w, h = struct.unpack('>II', head[16:24])
            return w, h

        # JPEG
        if head[:2] == b'\xff\xd8':
            f.seek(0)
            f.read(2)  # Skip SOI
            while True:
                marker, = struct.unpack('>H', f.read(2))
                if marker == 0xFFD9:  # EOI
                    break
                if marker == 0xFFDA:  # SOS
                    break
                length, = struct.unpack('>H', f.read(2))
                if 0xFFC0 <= marker <= 0xFFC3:  # SOF markers
                    f.read(1)  # precision
                    h, w = struct.unpack('>HH', f.read(4))
                    return w, h
                f.seek(length - 2, 1)
            return None, None

        # GIF
        if head[:6] in (b'GIF87a', b'GIF89a'):
            w, h = struct.unpack('<HH', head[6:10])
            return w, h

        # BMP
        if head[:2] == b'BM':
            w, h = struct.unpack('<II', head[18:26])
            return w, abs(h)

        # WebP
        if head[:4] == b'RIFF' and head[8:12] == b'WEBP':
            f.seek(0)
            data = f.read(100)
            # VP8
            vp8_idx = data.find(b'VP8 ')
            if vp8_idx != -1:
                w = (data[vp8_idx+14] | (data[vp8_idx+15] << 8)) & 0x3FFF
                h = (data[vp8_idx+16] | (data[vp8_idx+17] << 8)) & 0x3FFF
                return w, h
            # VP8L (lossless)
            vp8l_idx = data.find(b'VP8L')
            if vp8l_idx != -1:
                bits = struct.unpack('<I', data[vp8l_idx+9:vp8l_idx+13])[0]
                w = (bits & 0x3FFF) + 1
                h = ((bits >> 14) & 0x3FFF) + 1
                return w, h
            # VP8X (extended)
            vp8x_idx = data.find(b'VP8X')
            if vp8x_idx != -1:
                w = (data[vp8x_idx+12] | (data[vp8x_idx+13] << 8) | (data[vp8x_idx+14] << 16)) + 1
                h = (data[vp8x_idx+15] | (data[vp8x_idx+16] << 8) | (data[vp8x_idx+17] << 16)) + 1
                return w, h

    return None, None


def smart_resize(height, width, factor=28, min_pixels=56*56, max_pixels=14*14*4*1280):
    """
    Qwen2.5-VL official image resize logic (copied from transformers)
    Returns resized (height, width)
    """
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def calc_image_tokens(width, height, min_pixels, max_pixels, patch_size=28):
    """Calculate image token count"""
    h_bar, w_bar = smart_resize(height, width, factor=patch_size, min_pixels=min_pixels, max_pixels=max_pixels)
    return (h_bar // patch_size) * (w_bar // patch_size)


def init_worker(model_id, min_pixels, max_pixels, image_folder):
    """Worker initialization: only load tokenizer"""
    global _tokenizer, _image_folder, _min_pixels, _max_pixels
    from transformers import AutoTokenizer

    _image_folder = image_folder
    _min_pixels = min_pixels
    _max_pixels = max_pixels

    _tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    # Add LASER special tokens
    _tokenizer.add_tokens("<|laser_start|>", special_tokens=True)
    _tokenizer.add_tokens("<|laser|>", special_tokens=True)
    _tokenizer.add_tokens("<|laser_latent_end|>", special_tokens=True)
    _tokenizer.add_tokens("<|laser_end|>", special_tokens=True)


def process_sample(args):
    """Process a single sample"""
    global _tokenizer, _image_folder, _min_pixels, _max_pixels
    index, json_str = args

    try:
        sample = json.loads(json_str)

        # 1. Extract text
        conversations = sample.get("conversations", [])
        text_parts = []
        has_image = False

        for conv in conversations:
            value = conv.get("value", "")
            if "<image>" in value:
                has_image = True
                value = value.replace("<image>", "")
            text_parts.append(value)

        full_text = "\n".join(text_parts)

        # 2. Calculate text token count
        text_tokens = len(_tokenizer.encode(full_text, add_special_tokens=False))

        # 3. Calculate image token count
        image_tokens = 0
        image_paths = sample.get("image", [])
        if image_paths and has_image:
            img_path = image_paths[0] if isinstance(image_paths, list) else image_paths
            full_img_path = os.path.join(_image_folder, img_path)

            if os.path.exists(full_img_path):
                w, h = get_image_size_fast(full_img_path)
                if w and h:
                    image_tokens = calc_image_tokens(w, h, _min_pixels, _max_pixels)

        # 4. Total length = text + image + fixed overhead (chat template, special tokens, etc.)
        # Overhead estimate: <|im_start|>system...<|im_end|><|im_start|>user...<|im_end|><|im_start|>assistant
        # Approximately 30-50 tokens
        overhead = 50
        total_len = text_tokens + image_tokens + overhead

        return (index, total_len)

    except Exception as e:
        return (index, 0, str(e)[:100])


def load_existing(output_path):
    """Load existing progress"""
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r') as f:
                return {int(k): v for k, v in json.load(f).items()}
        except:
            pass
    return {}


def save_lengths(lengths, output_path):
    """Atomic write"""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    tmp = output_path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(lengths, f)
    os.replace(tmp, output_path)


def print_statistics(lengths):
    """Print statistics"""
    valid = [l for l in lengths.values() if l > 0]
    if not valid:
        print("No valid lengths!")
        return

    print(f"\n{'='*50}")
    print(f"Statistics: min={min(valid)}, max={max(valid)}, mean={sum(valid)/len(valid):.0f}")
    print(f"Valid: {len(valid)}, Invalid: {len(lengths) - len(valid)}")

    bins = [(0, 512), (512, 1024), (1024, 2048), (2048, 4096), (4096, 8192), (8192, 16384), (16384, float('inf'))]
    labels = ["<512", "512-1k", "1k-2k", "2k-4k", "4k-8k", "8k-16k", ">16k"]
    print("\nDistribution:")
    for (lo, hi), label in zip(bins, labels):
        cnt = sum(1 for l in valid if lo <= l < hi)
        pct = 100 * cnt / len(valid) if valid else 0
        print(f"  {label:>8}: {cnt:>6} ({pct:5.1f}%) {'#' * int(pct/2)}")


def sample_generator(data_path, existing_lengths):
    """Streaming generator"""
    with open(data_path, 'r') as f:
        data = json.load(f)

    for idx, sample in enumerate(data):
        if idx not in existing_lengths:
            yield (idx, json.dumps(sample, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description="Ultra-fast precompute sequence lengths")
    parser.add_argument("--data_path", type=str, required=True, help="Path to meta_data JSON")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--suffix", type=str, default="", help="Suffix to append to output filename (e.g., '_new')")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--min_pixels", type=int, default=128 * 28 * 28)
    parser.add_argument("--max_pixels", type=int, default=8192 * 28 * 28)
    parser.add_argument("--num_workers", type=int, default=128)
    parser.add_argument("--chunksize", type=int, default=500)
    parser.add_argument("--save_interval", type=int, default=10000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    # Apply suffix to output path
    output_path = args.output_path
    if args.suffix:
        base, ext = os.path.splitext(output_path)
        output_path = f"{base}{args.suffix}{ext}"

    print("Ultra-Fast Precompute Lengths (No Processor)")
    print("=" * 50)
    print(f"Model: {args.model_id}")
    print(f"Output: {output_path}")
    print(f"Workers: {args.num_workers}, Chunksize: {args.chunksize}")

    # Load existing progress
    all_lengths = {} if args.force else load_existing(output_path)
    if all_lengths:
        print(f"Loaded {len(all_lengths)} existing lengths")

    # Load meta_data
    print(f"\nLoading meta_data from {args.data_path}...")
    meta_data = json.load(open(args.data_path))

    meta = meta_data[0]
    data_path = meta['data_path']
    image_folder = meta['image_folder']

    # Get total count
    with open(data_path, 'r') as f:
        total_samples = len(json.load(f))

    pending = total_samples - len(all_lengths)
    print(f"Total: {total_samples}, Pending: {pending}")

    if pending <= 0:
        print("All done!")
        print_statistics(all_lengths)
        return

    # Parallel processing
    print(f"\nStarting with {args.num_workers} workers...")
    failed = 0
    computed = 0

    with multiprocessing.Pool(
        processes=args.num_workers,
        initializer=init_worker,
        initargs=(args.model_id, args.min_pixels, args.max_pixels, image_folder)
    ) as pool:
        pbar = tqdm(total=pending, desc="Processing")

        generator = sample_generator(data_path, all_lengths)

        for result in pool.imap_unordered(process_sample, generator, chunksize=args.chunksize):
            if len(result) == 2:
                idx, length = result
                all_lengths[idx] = length
            else:
                idx, length, err = result
                all_lengths[idx] = length
                failed += 1
                if failed <= 10:
                    tqdm.write(f"  [Error] idx={idx}: {err}")

            computed += 1
            pbar.update(1)

            if computed % args.save_interval == 0:
                save_lengths(all_lengths, output_path)
                pbar.set_postfix(saved=len(all_lengths), failed=failed)

        pbar.close()

    save_lengths(all_lengths, output_path)
    print(f"\nDone! Computed: {computed}, Failed: {failed}")
    print(f"Saved to {output_path}")
    print_statistics(all_lengths)


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()
