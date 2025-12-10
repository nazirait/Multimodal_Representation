import numpy as np
import matplotlib.pyplot as plt
import cv2


def plot_attention_map(att_map, tokens=None, title="Attention Map", figsize=(8,6), save_path=None):
    """Plot a 2D attention map (transformer head, etc)."""
    plt.figure(figsize=figsize)
    plt.imshow(att_map, cmap="viridis")
    plt.colorbar()
    if tokens is not None:
        plt.xticks(range(len(tokens)), tokens, rotation=90, fontsize=7)
        plt.yticks(range(len(tokens)), tokens, fontsize=7)
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

def plot_umap(latents, labels=None, title="UMAP Projection", save_path=None, palette='tab20', random_state=42):
    from umap import UMAP
    import scipy.sparse

    if "torch" in str(type(latents)):
        latents = latents.detach().cpu().numpy()

    umap = UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=random_state)
    emb = umap.fit_transform(latents)
    if isinstance(emb, tuple):
        emb = emb[0]
    if isinstance(emb, scipy.sparse.spmatrix):
        emb = emb.toarray() # type: ignore[attribute-error] 
    emb = np.asarray(emb)

    plt.figure(figsize=(8, 6))
    if labels is not None:
        num_classes = len(np.unique(labels))
        scatter = plt.scatter(
            emb[:, 0], emb[:, 1],
            c=labels,
            cmap=palette if num_classes <= 20 else "nipy_spectral",
            alpha=0.7,
            s=8
        )
        plt.colorbar(scatter, label='Label')
    else:
        plt.scatter(emb[:, 0], emb[:, 1], alpha=0.7, s=8)
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

def show_gradcam_on_image(img: np.ndarray, gradcam_map: np.ndarray, alpha=0.4, colormap=cv2.COLORMAP_JET, save_path=None):
    """
    Overlay a Grad-CAM heatmap on an image.
    img: np.ndarray, shape (H, W, 3), uint8 or float in [0, 1]
    gradcam_map: np.ndarray, shape (H, W), float in [0, 1]
    """
    if gradcam_map.ndim == 3:
        gradcam_map = gradcam_map.squeeze()
    gradcam_map = np.clip(gradcam_map, 0, 1)
    heatmap = np.uint8(255 * gradcam_map)
    colored_heatmap = cv2.applyColorMap(np.ascontiguousarray(heatmap, dtype=np.uint8), colormap)

    if img.dtype != np.uint8:
        img = np.clip(img, 0, 1)
        img = (img * 255).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if colored_heatmap.shape != img.shape:
        colored_heatmap = cv2.resize(colored_heatmap, (img.shape[1], img.shape[0]))
    overlay = cv2.addWeighted(img, 1 - alpha, colored_heatmap, alpha, 0)
    plt.figure(figsize=(6,6))
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    return overlay

def compute_gradcam(model, input_tensor, target_layer, target_category=None):
    """
    Wrapper for grad-cam.
    model: torch.nn.Module
    input_tensor: torch.Tensor, shape [1, 3, H, W]
    target_layer: torch.nn.Module, last conv layer
    target_category: int or None
    Returns: gradcam_map (H, W)
    """
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    # Move to correct device
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)

    cam = GradCAM(model=model, target_layers=[target_layer])
    targets = [ClassifierOutputTarget(target_category)] if target_category is not None else None
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0] #type:ignore [H, W], normalized
    return grayscale_cam
