
import torch
import itertools


''' slight modifications from OpenFold https://github.com/aqlaboratory/openfold/blob/f37d0d964d24b20494776b7f5643113dd584126c/openfold/data/data_transforms.py#L1183  '''



def crop_and_pad_to_fixed_size(
    protein: dict,
    shape_schema: dict,
    max_num_residues: int,
    seed: int = None,
) -> dict:
    """
    Applies random cropping and padding to ensure all tensors have uniform shape.

    Args:
        protein (dict): Dictionary of input tensors (from dataset).
        shape_schema (dict): Shape schema with "NUM_RES" for crop/pad dimensions.
        max_num_residues (int): Final number of residues to crop and pad to.
        seed (int, optional): Seed for reproducibility (e.g., per-protein). Default is None.

    Returns:
        dict: Cropped and padded protein dictionary.
    """
    cropped = random_crop_to_size(
        protein=protein,
        crop_size=max_num_residues,
        max_templates=0,
        shape_schema=shape_schema,
        seed=seed
    )

    padded = make_fixed_size(
        protein=cropped,
        shape_schema=shape_schema,
        num_residues=max_num_residues
    )

    return padded



def make_fixed_size(protein, shape_schema, num_residues=0):
    pad_size_map = {"NUM_RES":num_residues}

    for k, v in protein.items():
        # Don't transfer this to the accelerator.
        if k == "extra_cluster_assignment":
            continue
        shape = list(v.shape)
        schema = shape_schema[k]
        msg = "Rank mismatch between shape and shape schema for"
        assert len(shape) == len(schema), f"{msg} {k}: {shape} vs {schema}"
        pad_size = [
            pad_size_map.get(s2, None) or s1 for (s1, s2) in zip(shape, schema)
        ]

        padding = [(0, p - v.shape[i]) for i, p in enumerate(pad_size)]
        padding.reverse()
        padding = list(itertools.chain(*padding))
        if padding:
            protein[k] = torch.nn.functional.pad(v, padding)
            protein[k] = torch.reshape(protein[k], pad_size)
    
    return protein


def random_crop_to_size(
    protein,
    crop_size,
    max_templates,
    shape_schema,
    subsample_templates=False,
    seed=None,
    seq_length=None,
):
    """Crop randomly to `crop_size`, or keep as is if shorter than that."""
    
    # We want each ensemble to be cropped the same way

    g = None
    if seed is not None:
        g = torch.Generator(device=protein["aatype"].device)
        g.manual_seed(seed)

    if seq_length is None:
        seq_length = protein["aatype"].shape[0] 

    if "template_mask" in protein:
        num_templates = protein["template_mask"].shape[-1]
    else:
        num_templates = 0

    # No need to subsample templates if there aren't any
    subsample_templates = subsample_templates and num_templates

    num_res_crop_size = min(int(seq_length), crop_size)

    def _randint(lower, upper):
        return int(torch.randint(
                lower,
                upper + 1,
                (1,),
                device=protein["aatype"].device,
                generator=g,
        )[0])

    if subsample_templates:
        templates_crop_start = _randint(0, num_templates)
        templates_select_indices = torch.randperm(
            num_templates, device=protein["aatype"].device, generator=g
        )
    else:
        templates_crop_start = 0

    num_templates_crop_size = min(
        num_templates - templates_crop_start, max_templates
    )

    n = seq_length - num_res_crop_size
    if "use_clamped_fape" in protein and protein["use_clamped_fape"] == 1.:
        right_anchor = n
    else:
        x = _randint(0, n)
        right_anchor = n - x

    num_res_crop_start = _randint(0, right_anchor)

    for k, v in protein.items():
        if k not in shape_schema or (
            "template" not in k and "NUM_RES" not in shape_schema[k]
        ):
            continue

        # randomly permute the templates before cropping them.
        if k.startswith("template") and subsample_templates:
            v = v[templates_select_indices]

        slices = []
        for i, (dim_size, dim) in enumerate(zip(shape_schema[k], v.shape)):
            is_num_res = dim_size == "NUM_RES"
            if i == 0 and k.startswith("template"):
                crop_size = num_templates_crop_size
                crop_start = templates_crop_start
            else:
                crop_start = num_res_crop_start if is_num_res else 0
                crop_size = num_res_crop_size if is_num_res else dim
            slices.append(slice(crop_start, crop_start + crop_size))
        protein[k] = v[slices]

    # protein["seq_length"] = protein["seq_length"].new_tensor(num_res_crop_size)
    
    return protein