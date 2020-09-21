* @author  Dor Bernsohn
import cv2
import numpy as np

def refine_masks(masks, rois):
    """refine the mask to avoid overlapping

    Args:
        masks (array): array of (IMAGE_SIZE, IMAGE_SIZE, NUM_CLASSES) -> (1024, 1024, 11)
        rois (array): arrray of (NUM_CLASSES, [y_scale, x_scale, y_scale, x_scale])

    Returns:
        (array, array): masks, rois
    """     
    areas = np.sum(masks.reshape(-1, masks.shape[-1]), axis=0)
    mask_index = np.argsort(areas)
    union_mask = np.zeros(masks.shape[:-1], dtype=bool)
    for m in mask_index:
        masks[:, :, m] = np.logical_and(masks[:, :, m], np.logical_not(union_mask))
        union_mask = np.logical_or(masks[:, :, m], union_mask)
    for m in range(masks.shape[-1]):
        mask_pos = np.where(masks[:, :, m]==True)
        if np.any(mask_pos):
            y1, x1 = np.min(mask_pos, axis=1)
            y2, x2 = np.max(mask_pos, axis=1)
            rois[m, :] = [y1, x1, y2, x2]
    return masks, rois

def resize_image(image, config):
    """resize the image by the config settigs

    Args:
        image (array): array of (IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS) -> (1024, 1024, 3)
        config (config): config class of the dataset

    Returns:
        array: array of (config.IMAGE_MAX_DIM, config.IMAGE_MAX_DIM, NUM_CHANNELS) -> (1024, 1024, 3)
    """    
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (config.IMAGE_MAX_DIM, config.IMAGE_MAX_DIM), interpolation=cv2.INTER_AREA)  
    return img