from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from segmentation.segmenter import *
from segmentation.well_mask import WellMask
from segmentation.mask_container import MaskContainer


def cluster_area(mc: "MaskContainer", eps: int , min_samples: int = 3, debug: bool = False) -> tuple["MaskContainer", "MaskContainer"]:
    areas = np.array([m.area for m in mc.masks]).reshape(-1, 1)
    coords = np.array([(m.center[1], m.center[0]) for m in mc.masks]) # xy, used for plotting
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(areas)
    labels = clustering.labels_  # -1 is noise

    unique_area_labels = list(set(labels))

    cluster_sizes = []  # Store (label, size)
    result_mcs = []
    
    for area_label in unique_area_labels:
        mask_group = [mc.masks[idx] for idx, label in enumerate(labels) if label == area_label]
        cluster_sizes.append((area_label, len(mask_group)))
        result_mcs.append(MaskContainer(image=mc.image, masks=mask_group))
    
    # Sort clusters by number of elements descending
    cluster_sizes.sort(key=lambda x: x[1], reverse=True)
    
    # Find cluster with the most elements
    largest_cluster_label = cluster_sizes[0][0]

    # Create two groups
    large_masks = [mc.masks[idx] for idx, label in enumerate(labels) if int(label) == largest_cluster_label]
    large_mc = MaskContainer(image=mc.image, masks=large_masks)

    if debug:
        plt.figure(figsize=(10, 10))
        plt.imshow(mc.image, cmap='gray')
        colors = plt.cm.tab10(np.linspace(0, 1, len(set(labels))))
        for idx, (x, y) in enumerate(coords):
            label = labels[idx]
            color = 'red' if label == -1 else colors[label % len(colors)]
            plt.plot(x, y, 'o', color=color, markersize=5)
    
        plt.title("DBSCAN Area Results")
        plt.axis('off')
        plt.show()
        print(f"==== removed {len(mc) - len(large_mc)} masks ====")
    
    return large_mc, result_mcs


def cluster_coords(mc: "MaskContainer", eps: int, min_samples: int = 3, debug: bool = False) -> "MaskContainer":
    coords = np.array([(m.center[1], m.center[0]) for m in mc.masks]) # xy

    # Run DBSCAN on xy_coords
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    labels = clustering.labels_  # -1 is noise
    
    unique_labels = list(dict.fromkeys(labels))
    result_masks = []

    for label in unique_labels:
        mask_group = [mc.masks[idx] for idx, l in enumerate(labels) if l == label and label != -1]
        result_masks += mask_group

    if debug:
        plt.figure(figsize=(10, 10))
        plt.imshow(mc.image, cmap='gray')
        colors = plt.cm.tab10(np.linspace(0, 1, len(set(labels))))
        
        for idx, (x, y) in enumerate(coords):
            label = labels[idx]
            color = 'red' if label == -1 else colors[label % len(colors)]
            plt.plot(x, y, 'o', color=color, markersize=5)
        
        plt.title("DBSCAN Results")
        plt.axis('off')
        plt.show()
        print(f"==== removed {len(mc) - len(result_masks)} masks ====")

    return MaskContainer(image=mc.image, masks=result_masks)