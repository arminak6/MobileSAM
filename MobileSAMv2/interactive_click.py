import argparse
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from mobilesamv2.promt_mobilesamv2 import ObjectAwareModel
from mobilesamv2 import sam_model_registry, SamPredictor

# LoRA / PEFT
from peft import LoraConfig, get_peft_model


ENCODER_PATH = {
    "efficientvit_l2": "./weight/l2.pt",
    "tiny_vit": "./weight/mobile_sam.pt",
    "sam_vit_h": "./weight/sam_vit_h.pt",
}


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--img", type=str, required=True, help="single image path (e.g. ./test_images/1.jpg)")
    p.add_argument("--output", type=str, default="./out.png", help="output png path")
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--iou", type=float, default=0.9)
    p.add_argument("--conf", type=float, default=0.4)
    p.add_argument("--retina", type=bool, default=True)

    # IMPORTANT: default to tiny_vit (matches your LoRA training)
    p.add_argument(
        "--encoder_type",
        choices=["tiny_vit", "sam_vit_h", "efficientvit_l2"],
        default="tiny_vit",
    )

    p.add_argument("--ObjectAwareModel_path", type=str, default="./weight/ObjectAwareModel.pt")
    p.add_argument("--Prompt_guided_Mask_Decoder_path", type=str,
                   default="./PromptGuidedDecoder/Prompt_guided_Mask_Decoder.pt")

    # Your trained weights
    p.add_argument("--lora_weights", type=str, default="lora_weights.pt",
                   help="Saved from: torch.save(model.image_encoder.state_dict(), ...)")
    p.add_argument("--decoder_weights", type=str, default="trained_decoder.pt",
                   help="Saved from: torch.save(model.mask_decoder.state_dict(), ...)")

    # LoRA hyperparams (defaults match YOUR training script)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_bias", type=str, default="none", choices=["none", "all", "lora_only"])
    p.add_argument("--lora_target_modules", type=str, default="qkv",
                   help="Must match training. For your training it was: qkv")

    p.add_argument("--merge_lora", action="store_true",
                   help="Merge LoRA into base encoder weights after loading (faster inference)")

    return p.parse_args()


def _split_csv(s: str):
    return [x.strip() for x in s.split(",") if x.strip()]


def _safe_torch_load(path: str, device: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Not found: {path}")
    # Torch 2.4+ warning: weights_only default may change. We keep simple and compatible.
    return torch.load(path, map_location=device)


def _strip_prefix_from_state_dict(sd: dict, prefixes):
    """Remove common prefixes if present (e.g., 'module.', 'base_model.model.')."""
    if not isinstance(sd, dict):
        return sd
    new_sd = {}
    for k, v in sd.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p):]
        new_sd[nk] = v
    return new_sd


def _load_state_dict_with_report(module: nn.Module, path: str, device: str, name: str, strict: bool = False):
    sd = _safe_torch_load(path, device)
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]

    # Clean typical prefixes
    sd = _strip_prefix_from_state_dict(sd, prefixes=["module.", "base_model.model.", "model."])

    # Count matching keys (useful to ensure something actually loaded)
    model_keys = set(module.state_dict().keys())
    sd_keys = set(sd.keys())
    matched = len(model_keys.intersection(sd_keys))

    out = module.load_state_dict(sd, strict=strict)

    # torch returns NamedTuple with missing/unexpected in newer versions; in older versions it returns None
    missing = getattr(out, "missing_keys", [])
    unexpected = getattr(out, "unexpected_keys", [])

    print(f"[OK] Loaded {name} from: {path}")
    print(f"     matched keys: {matched} / {len(model_keys)} (strict={strict})")
    if missing:
        print(f"[WARN] {name} missing keys (first 20): {missing[:20]}")
    if unexpected:
        print(f"[WARN] {name} unexpected keys (first 20): {unexpected[:20]}")


def create_model(args):
    obj_model = ObjectAwareModel(args.ObjectAwareModel_path)

    PromptGuidedDecoder = sam_model_registry["PromptGuidedDecoder"](args.Prompt_guided_Mask_Decoder_path)

    # Keep your original assembly style (same as your training)
    model = sam_model_registry["vit_h"]()
    model.prompt_encoder = PromptGuidedDecoder["PromtEncoder"]
    model.mask_decoder = PromptGuidedDecoder["MaskDecoder"]

    image_encoder = sam_model_registry[args.encoder_type](ENCODER_PATH[args.encoder_type])
    model.image_encoder = image_encoder

    return model, obj_model


class ClickState:
    def __init__(self):
        self.pos = []
        self.neg = []
        self.box = None
        self._drag = False
        self._start = None


def draw_overlay(img_bgr, st: ClickState):
    vis = img_bgr.copy()

    if st.box is not None:
        x0, y0, x1, y1 = st.box
        cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 255, 0), 2)

    for (x, y) in st.pos:
        cv2.circle(vis, (x, y), 5, (0, 255, 0), -1)
    for (x, y) in st.neg:
        cv2.circle(vis, (x, y), 5, (0, 0, 255), -1)

    cv2.putText(
        vis,
        "Left=+  Right=-  SHIFT+drag=box  ENTER=segment  R=reset  Q=quit",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    return vis


def mouse_cb(event, x, y, flags, st: ClickState):
    # SHIFT + drag -> box
    if (flags & cv2.EVENT_FLAG_SHIFTKEY) and event == cv2.EVENT_LBUTTONDOWN:
        st._drag = True
        st._start = (x, y)
        st.box = (x, y, x, y)

    if st._drag and event == cv2.EVENT_MOUSEMOVE and st._start is not None:
        x0, y0 = st._start
        st.box = (min(x0, x), min(y0, y), max(x0, x), max(y0, y))

    if event == cv2.EVENT_LBUTTONUP and st._drag:
        st._drag = False
        st._start = None

    # normal clicks
    if not (flags & cv2.EVENT_FLAG_SHIFTKEY):
        if event == cv2.EVENT_LBUTTONDOWN:
            st.pos.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            st.neg.append((x, y))


def pick_largest_box(obj_results):
    boxes = obj_results[0].boxes.xyxy
    if boxes is None or len(boxes) == 0:
        return None
    b = boxes.detach().cpu().numpy().astype(np.float32)
    areas = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return b[int(np.argmax(areas))]


def run_interactive(image_rgb, predictor: SamPredictor, obj_model, device, args):
    obj_results = obj_model(
        image_rgb,
        device=device,
        retina_masks=args.retina,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
    )
    default_box = pick_largest_box(obj_results)

    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    st = ClickState()

    win = "MobileSAMv2 Interactive"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, lambda e, x, y, f, p: mouse_cb(e, x, y, f, st))

    while True:
        cv2.imshow(win, draw_overlay(img_bgr, st))
        k = cv2.waitKey(20) & 0xFF

        if k in (ord("q"), 27):
            cv2.destroyWindow(win)
            return None

        if k == ord("r"):
            st.pos.clear()
            st.neg.clear()
            st.box = None

        if k in (13, 10):  # Enter
            box_xyxy = None
            if st.box is not None:
                box_xyxy = np.array(st.box, dtype=np.float32)
            elif default_box is not None:
                box_xyxy = default_box

            pts = st.pos + st.neg
            if len(pts) == 0 and box_xyxy is None:
                print("Add at least one point or draw a box (SHIFT+drag).")
                continue

            points_t = None
            if len(pts) > 0:
                point_coords = np.array(pts, dtype=np.float32)
                point_labels = np.array([1] * len(st.pos) + [0] * len(st.neg), dtype=np.int64)

                pc_t = predictor.transform.apply_coords(point_coords, predictor.original_size)
                pc_t = torch.as_tensor(pc_t, dtype=torch.float32, device=device)[None, :, :]
                pl_t = torch.as_tensor(point_labels, dtype=torch.int64, device=device)[None, :]
                points_t = (pc_t, pl_t)

            boxes_t = None
            if box_xyxy is not None:
                bt = predictor.transform.apply_boxes(box_xyxy[None, :], predictor.original_size)
                boxes_t = torch.as_tensor(bt, dtype=torch.float32, device=device)

            with torch.no_grad():
                image_embedding = predictor.features
                prompt_pe = predictor.model.prompt_encoder.get_dense_pe()

                sparse_embeddings, dense_embeddings = predictor.model.prompt_encoder(
                    points=points_t,
                    boxes=boxes_t,
                    masks=None,
                )

                low_res_masks, _ = predictor.model.mask_decoder(
                    image_embeddings=image_embedding,
                    image_pe=prompt_pe,
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                    simple_type=True,
                )

                masks = predictor.model.postprocess_masks(
                    low_res_masks, predictor.input_size, predictor.original_size
                )
                mask = (masks > predictor.model.mask_threshold).float()[0, 0].cpu().numpy()

            cv2.destroyWindow(win)
            return mask


def save_mask_viz(image_rgb, mask, out_path):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)
    plt.imshow(mask, alpha=0.5)
    plt.axis("off")
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.0)
    plt.close()


def apply_lora_to_encoder(image_encoder, args):
    target_modules = _split_csv(args.lora_target_modules)
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias,
        target_modules=target_modules,
    )
    return get_peft_model(image_encoder, lora_cfg)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    print(f"[INFO] encoder_type={args.encoder_type} (professional: must match training)")
    print(f"[INFO] LoRA target_modules={args.lora_target_modules} r={args.lora_r} alpha={args.lora_alpha} dropout={args.lora_dropout}")

    img_bgr = cv2.imread(args.img)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {args.img}")
    image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Build model like training
    model, obj_model = create_model(args)

    # Apply LoRA wrapper BEFORE loading lora_weights.pt
    print("[INFO] Applying LoRA wrapper to image encoder...")
    model.image_encoder = apply_lora_to_encoder(model.image_encoder, args)

    # Load weights
    print("[INFO] Loading weights...")
    _load_state_dict_with_report(model.image_encoder, args.lora_weights, device, name="image_encoder(LoRA)", strict=False)
    _load_state_dict_with_report(model.mask_decoder, args.decoder_weights, device, name="mask_decoder", strict=False)

    # Optionally merge LoRA (faster inference)
    if args.merge_lora:
        print("[INFO] Merging LoRA into base weights...")
        model.image_encoder = model.image_encoder.merge_and_unload()

    model.to(device)
    model.eval()

    predictor = SamPredictor(model)
    predictor.set_image(image_rgb)

    mask = run_interactive(image_rgb, predictor, obj_model, device, args)
    if mask is None:
        print("Quit without segmenting.")
        return

    save_mask_viz(image_rgb, mask, args.output)
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
